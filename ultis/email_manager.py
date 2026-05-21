import imaplib
import smtplib
import email
import mimetypes
from email.header import decode_header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.header import Header
from email import encoders
import os
import ssl
import time
import re
import base64
import traceback
from datetime import datetime
from typing import Union, List, Optional, Tuple

class EmailManager:
    """腾讯企业邮箱管理类，支持IMAP下载附件、SMTP发送邮件"""

    # 支持的图片扩展名
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}

    def __init__(self, email: str = "pagududeshengjiang@shpbjj.com",
                 auth_code: str = "RyizdLkyE2onVbdY",
                 imap_server: str = "imap.exmail.qq.com",
                 smtp_server: str = "smtp.exmail.qq.com",
                 save_dir: str = "/home/trading/招商DMA交易表"):
        """
        初始化邮箱管理器
        :param email: 邮箱地址
        :param auth_code: 客户端专用密码（授权码）
        :param imap_server: IMAP服务器地址
        :param smtp_server: SMTP服务器地址
        :param save_dir: 附件保存目录
        """
        self.email = email
        self.auth_code = auth_code
        self.imap_server = imap_server
        self.smtp_server = smtp_server
        self.save_dir = save_dir
        self.imap_conn = None
        self.smtp_conn = None

        # 创建保存目录
        # os.makedirs(self.save_dir, exist_ok=True)
        # print(f"✅ 附件保存目录: {os.path.abspath(self.save_dir)}")

    def _create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """创建宽松的SSL上下文（用于处理证书问题）"""
        try:
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            return context
        except:
            return None

    def connect_imap(self) -> bool:
        """连接IMAP服务器并登录"""
        try:
            context = self._create_ssl_context()
            if context:
                self.imap_conn = imaplib.IMAP4_SSL(self.imap_server, 993, ssl_context=context)
            else:
                self.imap_conn = imaplib.IMAP4_SSL(self.imap_server, 993)
            print(f"✅ IMAP连接建立")

            self.imap_conn.login(self.email, self.auth_code)
            print(f"✅ 已登录邮箱: {self.email}")
            return True
        except imaplib.IMAP4.error as e:
            print(f"❌ IMAP登录失败: {e}")
            self._handle_imap_error(e)
            return False
        except Exception as e:
            print(f"❌ IMAP连接异常: {e}")
            traceback.print_exc()
            return False

    def _handle_imap_error(self, error: Exception):
        """处理IMAP常见错误"""
        error_msg = str(error)
        if "Unsafe Login" in error_msg or "authentication failed" in error_msg.lower():
            print("\n⚠️ 腾讯企业邮箱安全限制解决方案：")
            print("1. 登录腾讯企业邮箱网页版 (exmail.qq.com)")
            print("2. 进入【设置】→【客户端专用密码】")
            print("3. 生成新的客户端专用密码（授权码）")
            print("4. 确保IMAP/SMTP服务已开启")
            print("5. 保存设置后等待几分钟")
            print("6. 重新运行此脚本，并使用生成的客户端专用密码")

    def _select_folder(self, folder_name: str = 'INBOX') -> bool:
        """选择邮箱文件夹"""
        if not self.imap_conn:
            return False

        # 尝试直接选择
        status, data = self.imap_conn.select(folder_name, readonly=False)
        if status == 'OK':
            print(f"✅ 已选择文件夹: {folder_name}")
            return True

        # 如果INBOX不存在，尝试查找中文"收件箱"
        if folder_name == 'INBOX':
            typ, folders = self.imap_conn.list()
            if typ == 'OK':
                for folder in folders:
                    folder_name_raw = folder.decode().split('"')[-2]
                    try:
                        decoded = self._imap_utf7_decode(folder_name_raw)
                        if decoded == '收件箱':
                            status, data = self.imap_conn.select(folder_name_raw, readonly=False)
                            if status == 'OK':
                                print(f"✅ 已选择文件夹: {folder_name_raw}")
                                return True
                    except:
                        pass
        print(f"❌ 选择文件夹 '{folder_name}' 失败")
        return False

    def _imap_utf7_decode(self, s: str) -> str:
        """简单实现IMAP UTF-7解码（仅用于解码文件夹名）"""
        result = []
        i = 0
        while i < len(s):
            if s[i] == '&':
                j = s.find('-', i + 1)
                if j == -1:
                    result.append(s[i:])
                    break
                if j == i + 1:
                    result.append('&')
                else:
                    encoded = s[i + 1:j].replace(',', '/')
                    try:
                        decoded = base64.b64decode(encoded).decode('utf-16-be')
                        result.append(decoded)
                    except:
                        result.append(s[i:j + 1])
                i = j + 1
            else:
                result.append(s[i])
                i += 1
        return ''.join(result)

    def search_emails(self, keywords: Optional[list] = None, limit: int = 10) -> List[bytes]:
        """
        搜索邮件（按主题包含关键字）
        :param keyword: 主题关键字，为None则搜索全部
        :param limit: 返回的最大邮件数量（最新）
        :return: 邮件ID列表
        """
        if not self.imap_conn:
            if not self.connect_imap():
                return []

        if not self._select_folder('INBOX'):
            return []

        # 构建搜索条件
        if keywords:
            # 搜索主题包含关键字（需先获取邮件头，此方法效率较低，但简单）
            # 注意：IMAP的SEARCH命令不支持直接按主题模糊搜索，这里采用拉取所有邮件后过滤
            # 为提高性能，也可使用CRITERIA，但简单起见用全量+过滤
            status, messages = self.imap_conn.search(None, 'ALL')
        else:
            status, messages = self.imap_conn.search(None, 'ALL')

        if status != 'OK':
            print(f"❌ 搜索邮件失败: {messages}")
            return []

        email_ids = messages[0].split()
        if not email_ids:
            return []

        # 获取最新的limit封
        if limit > 0:
            email_ids = email_ids[-limit:]

        # 如果有关键字，过滤主题
        if keywords:
            filtered_ids = []
            for eid in email_ids:
                status, msg_data = self.imap_conn.fetch(eid, '(BODY.PEEK[HEADER.FIELDS (SUBJECT)])')
                if status != 'OK':
                    continue
                # 解析主题
                subject_header = msg_data[0][1].decode('utf-8', errors='ignore')
                subject = self._parse_header_field(subject_header, 'Subject')
                if all(keyword.lower() in subject.lower() for keyword in keywords):
                    filtered_ids.append(eid)
            
            return filtered_ids
        else:
            return email_ids

    def _parse_header_field_1(self, header_data: str, field_name: str) -> str:
        """从邮件头数据中解析指定字段的值"""
        for line in header_data.split('\n'):
            if line.lower().startswith(field_name.lower() + ':'):
                value = line.split(':', 1)[1].strip()
                # 解码
                decoded = decode_header(value)
                result = []
                for part, enc in decoded:
                    if isinstance(part, bytes):
                        part = part.decode(enc or 'utf-8', errors='ignore')
                    result.append(part)
                return ''.join(result)
        return ""

    def _parse_header_field(self, header_data: str, field_name: str) -> str:
        """从邮件头数据中解析指定字段的值，特别处理多行中文主题"""
        # 提取主题字段的完整值（包括多行）
        subject_lines = []
        for line in header_data.split('\n'):
            if line.lower().startswith(field_name.lower() + ':'):
                # 获取主题字段的值
                value = line.split(':', 1)[1].strip()
                subject_lines.append(value)
            elif line.startswith(' '):
                # 处理多行主题的后续行（以空格开头）
                subject_lines.append(line.strip())
        
        if not subject_lines:
            return ""
        
        # 合并多行主题
        full_subject = ' '.join(subject_lines)
        
        # 尝试解码
        try:
            # 使用 email.header.decode_header 自动处理多行编码
            decoded = decode_header(full_subject)
            result = []
            for part, enc in decoded:
                if isinstance(part, bytes):
                    # 优先尝试 UTF-8 解码
                    try:
                        part = part.decode('utf-8', errors='ignore')
                    except:
                        # 备用解码方式
                        part = part.decode(enc or 'latin1', errors='ignore')
                result.append(part)
            return ''.join(result)
        except Exception as e:
            # 如果解码失败，返回原始值
            return full_subject

    def download_attachments_1(self, email_id: bytes, save_dir: str,
                             file_extensions: Optional[List[str]] = None) -> int:
        """
        下载指定邮件的附件
        :param email_id: 邮件ID
        :param save_dir: 保存目录
        :param file_extensions: 允许的扩展名列表，如['.xlsx','.jpg']，None表示所有
        :return: 下载的附件数量
        """
        if not self.imap_conn:
            return 0

        status, msg_data = self.imap_conn.fetch(email_id, '(RFC822)')
        if status != 'OK':
            print(f"❌ 获取邮件内容失败")
            return 0

        msg = email.message_from_bytes(msg_data[0][1])

        # 解析主题用于日志
        subject = self._parse_header_field_from_msg(msg, 'Subject')
        print(f"📨 处理邮件: {subject}")

        count = 0
        for part in msg.walk():
            if part.get_content_maintype() == 'multipart':
                continue

            filename = part.get_filename()
            if not filename:
                continue

            # 解码文件名
            filename = decode_header(filename)[0][0]
            if isinstance(filename, bytes):
                filename = filename.decode('utf-8', errors='ignore')

            # 检查扩展名
            ext = os.path.splitext(filename)[1].lower()
            if file_extensions is not None and ext not in file_extensions:
                continue

            # 清理文件名
            clean_name = re.sub(r'[\\/*?:"<>|]', "", filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_filename = f"{timestamp}_{clean_name}"
            filepath = os.path.join(save_dir, clean_name)

            # 保存附件
            with open(filepath, 'wb') as f:
                f.write(part.get_payload(decode=True))

            print(f"📥 已下载附件: {clean_name}")
            count += 1

        return count

    def download_attachments(self, email_id: bytes, save_dir: str,
                            file_extensions: Optional[List[str]] = None) -> int:
        """
        下载指定邮件的附件
        :param email_id: 邮件ID
        :param save_dir: 保存目录
        :param file_extensions: 允许的扩展名列表，如['.xlsx','.jpg']，None表示所有
        :return: 下载的附件数量
        """
        if not self.imap_conn:
            return 0

        # 创建目录（如果不存在）
        os.makedirs(save_dir, exist_ok=True)

        status, msg_data = self.imap_conn.fetch(email_id, '(RFC822)')
        if status != 'OK':
            print(f"❌ 获取邮件内容失败")
            return 0

        msg = email.message_from_bytes(msg_data[0][1])

        subject = self._parse_header_field_from_msg(msg, 'Subject')
        print(f"📨 处理邮件: {subject}")

        count = 0
        for part in msg.walk():
            # 只关心 attachment 类型的 part
            if part.get_content_disposition() != 'attachment':
                continue

            filename = part.get_filename()
            if not filename:
                continue

            # 解码文件名
            decoded_parts = decode_header(filename)
            filename_str = ''
            for content, charset in decoded_parts:
                if isinstance(content, bytes):
                    try:
                        # 优先尝试UTF-8解码
                        filename_str += content.decode('utf-8', errors='ignore')
                    except:
                        # 备用解码方式
                        filename_str += content.decode(charset or 'latin1', errors='ignore')
                else:
                    filename_str += content
        
            # 检查扩展名
            ext = os.path.splitext(filename_str)[1].lower()
            if file_extensions is not None and ext not in file_extensions:
                continue

            # 清理非法字符
            clean_name = re.sub(r'[\\/*?:"<>|]', '', filename_str)
            clean_name = re.sub(r'_d(?=\.|$)', '', clean_name)
            filepath = os.path.join(save_dir, clean_name)  

            # 保存附件
            with open(filepath, 'wb') as f:
                f.write(part.get_payload(decode=True))

            print(f"📥 已下载附件: {clean_name} -> {clean_name}")
            count += 1

        return count

    def _parse_header_field_from_msg(self, msg, field_name: str) -> str:
        """从email.message对象中解析头部字段"""
        header = msg.get(field_name, '')
        if not header:
            return ''
        decoded = decode_header(header)
        result = []
        for part, enc in decoded:
            if isinstance(part, bytes):
                part = part.decode(enc or 'utf-8', errors='ignore')
            result.append(part)
        return ''.join(result)

    def download_attachments_by_keyword(self, keyword: Union[List[str], str] = None, save_dir: Optional[str] = None,
                                        file_extensions: Optional[List[str]] = None,
                                        limit: int = 50) -> int:
        """
        根据主题关键字下载附件
        :param keyword: 主题关键字
        :param save_dir: 保存目录（默认使用self.save_dir）
        :param file_extensions: 允许的扩展名列表
        :param limit: 最多处理邮件数量
        :return: 下载的附件总数
        """
        if save_dir is None:
            save_dir = self.save_dir

        if keyword is None:
            print(f"⚠️ 未传入关键字")
            return 0

        if isinstance(keyword, str):
            keyword = [keyword]

        email_ids = self.search_emails(keyword, limit)
        if not email_ids:
            print(f"⚠️ 未找到包含关键字 {keyword} 的邮件")
            return 0

        total = 0
        for eid in email_ids:
            total += self.download_attachments(eid, save_dir, file_extensions)
        print(f"🎉 共下载 {total} 个附件（关键字：{keyword}, 路径:{save_dir}）")
        return total

    def download_images_by_keyword(self, keyword: str, save_dir: Optional[str] = None,
                                   limit: int = 10) -> int:
        """
        根据主题关键字下载图片附件
        :param keyword: 主题关键字
        :param save_dir: 保存目录（默认使用self.save_dir）
        :param limit: 最多处理邮件数量
        :return: 下载的图片数量
        """
        return self.download_attachments_by_keyword(keyword, save_dir,
                                                    file_extensions=list(self.IMAGE_EXTENSIONS),
                                                    limit=limit)

    def connect_smtp(self) -> bool:
        """连接SMTP服务器并登录"""
        try:
            self.smtp_conn = smtplib.SMTP_SSL(self.smtp_server, 465)
            self.smtp_conn.login(self.email, self.auth_code)
            print(f"✅ SMTP登录成功")
            return True
        except Exception as e:
            print(f"❌ SMTP连接失败: {e}")
            return False

    def send_email_with_attachments(self, to: list, subject: str, body: str,
                                    attachments: Optional[List[str]] = None,
                                    is_html: bool = False) -> bool:
        """
        发送带附件的邮件
        :param to: 收件人邮箱
        :param subject: 主题
        :param body: 正文
        :param attachments: 附件文件路径列表
        :param is_html: 正文是否为HTML格式
        :return: 发送成功与否
        """
        if not self.smtp_conn:
            if not self.connect_smtp():
                return False

        msg = MIMEMultipart()
        msg['From'] = self.email
        if len(to) == 0:
            act_to = self.email
        elif len(to) < 2:
            act_to = to[0]
        else:
            act_to = ', '.join(to)
        msg['To'] = act_to
        msg['Subject'] = subject

        # 添加正文
        content_type = 'html' if is_html else 'plain'
        msg.attach(MIMEText(body, content_type, 'utf-8'))

        # 添加附件
        if attachments:
            for file_path in attachments:
                if not os.path.isfile(file_path):
                    print(f"⚠️ 附件不存在: {file_path}")
                    continue
                try:
                    with open(file_path, 'rb') as f:
                        # part = MIMEBase('application', 'octet-stream')
                        # part.set_payload(f.read())
                        # encoders.encode_base64(part)
                        # filename = os.path.basename(file_path)
                        # part.add_header('Content-Disposition', f'attachment; filename="{filename}"')
                        # --- 修改开始 ---
                    
                        # 1. 自动识别 MIME 类型 (例如识别出 application/pdf)
                        #    如果无法识别，则默认使用 octet-stream
                        mime_type, _ = mimetypes.guess_type(file_path)
                        if mime_type is None:
                            mime_type = 'application/octet-stream'
                        
                        # 分割主类型和子类型 (例如 'application' 和 'pdf')
                        main_type, sub_type = mime_type.split('/', 1)
                        
                        # 2. 使用识别出的类型创建 MIMEBase，而不是强制使用 octet-stream
                        part = MIMEBase(main_type, sub_type)
                        part.set_payload(f.read())
                        encoders.encode_base64(part)
                        
                        # 3. 规范文件名编码，防止中文乱码或被识别为 bin
                        filename = os.path.basename(file_path)
                        # 使用 Header 对象进行编码，兼容性更好
                        part.add_header('Content-Disposition', 'attachment', filename=Header(filename, 'utf-8').encode())
                        
                        # --- 修改结束 ---
                        msg.attach(part)
                except Exception as e:
                    print(f"❌ 添加附件失败 {file_path}: {e}")

        try:
            self.smtp_conn.send_message(msg)
            print(f"✅ 邮件已发送至 {to}")
            return True
        except Exception as e:
            print(f"❌ 发送邮件失败: {e}")
            return False

    def send_email_with_inline_images(self, to: str, subject: str, body_html: str,
                                      image_paths: List[str]) -> bool:
        """
        发送HTML邮件，内嵌图片（使用cid）
        :param to: 收件人
        :param subject: 主题
        :param body_html: HTML正文，其中图片src需为 cid:image_cid
        :param image_paths: 图片文件路径列表，将自动分配cid
        :return: 发送成功与否
        """
        if not self.smtp_conn:
            if not self.connect_smtp():
                return False

        msg = MIMEMultipart('related')
        msg['From'] = self.email
        msg['To'] = to
        msg['Subject'] = subject

        # 构造HTML部分
        html_part = MIMEText(body_html, 'html', 'utf-8')
        msg.attach(html_part)

        # 添加内嵌图片
        for i, img_path in enumerate(image_paths):
            if not os.path.isfile(img_path):
                print(f"⚠️ 图片不存在: {img_path}")
                continue
            try:
                with open(img_path, 'rb') as f:
                    img_data = f.read()
                # 获取图片MIME类型
                ext = os.path.splitext(img_path)[1].lower()
                if ext in ['.jpg', '.jpeg']:
                    mime = 'image/jpeg'
                elif ext == '.png':
                    mime = 'image/png'
                elif ext == '.gif':
                    mime = 'image/gif'
                else:
                    mime = 'application/octet-stream'

                img_part = MIMEBase(mime.split('/')[0], mime.split('/')[1])
                img_part.set_payload(img_data)
                encoders.encode_base64(img_part)
                cid = f"image_{i}"  # 或使用文件名
                img_part.add_header('Content-ID', f'<{cid}>')
                img_part.add_header('Content-Disposition', 'inline', filename=os.path.basename(img_path))
                msg.attach(img_part)
            except Exception as e:
                print(f"❌ 添加内嵌图片失败 {img_path}: {e}")

        try:
            self.smtp_conn.send_message(msg)
            print(f"✅ 内嵌图片邮件已发送至 {to}")
            return True
        except Exception as e:
            print(f"❌ 发送邮件失败: {e}")
            return False


    def send_email_with_images_and_attachments(self, to: list, subject: str,
                                                body_html: Optional[str] = None,
                                                image_paths: Optional[List[str]] = None,
                                                attachments: Optional[List[str]] = None) -> bool:
        """
        Send email with inline images (displayed in body) AND file attachments simultaneously.

        - If body_html is provided, it can reference images via cid:image_0, cid:image_1, etc.
        - If body_html is None and image_paths is provided, a default HTML body with
          all images displayed in order will be auto-generated.

        :param to: recipient list
        :param subject: email subject
        :param body_html: optional HTML body; if None and image_paths given, auto-generated
        :param image_paths: list of image file paths for inline display
        :param attachments: list of file paths to attach
        :return: True if sent successfully
        """
        if not self.smtp_conn:
            if not self.connect_smtp():
                return False

        # Auto-generate HTML body if not provided
        if not body_html and image_paths:
            img_tags = []
            for i in range(len(image_paths)):
                img_tags.append(
                    '<p><img src="cid:image_%d" style="max-width:100%%; height:auto; '
                    'display:block; margin:10px auto;" alt="image_%d"></p>' % (i, i)
                )
            body_html = (
                '<!DOCTYPE html>\n'
                '<html>\n'
                '<head><meta charset="utf-8"></head>\n'
                '<body style="font-family:\'Microsoft YaHei\', SimHei, Arial, sans-serif; padding:20px;">\n'
                '    <h2 style="color:#2C3E50;">%s</h2>\n'
                '%s\n'
                '    <hr style="border:none; border-top:1px solid #eee; margin:20px 0;">\n'
                '    <p style="color:#999; font-size:12px;">Generated by EmailManager</p>\n'
                '</body>\n'
                '</html>'
            ) % (subject, '\n'.join(img_tags))

        if not body_html:
            body_html = '<p>No content.</p>'

        # Outer: mixed (supports both inline and attachments)
        msg = MIMEMultipart('mixed')
        msg['From'] = self.email
        if len(to) == 0:
            act_to = self.email
        elif len(to) < 2:
            act_to = to[0]
        else:
            act_to = ', '.join(to)
        msg['To'] = act_to
        msg['Subject'] = subject

        # Inner: related (HTML + inline images)
        related = MIMEMultipart('related')

        # Attach HTML body
        related.attach(MIMEText(body_html, 'html', 'utf-8'))

        # Attach inline images
        if image_paths:
            for i, img_path in enumerate(image_paths):
                if not os.path.isfile(img_path):
                    print(f"[WARN] Image not found: {img_path}")
                    continue
                try:
                    with open(img_path, 'rb') as f:
                        img_data = f.read()
                    ext = os.path.splitext(img_path)[1].lower()
                    if ext in ('.jpg', '.jpeg'):
                        mime = 'image/jpeg'
                    elif ext == '.png':
                        mime = 'image/png'
                    elif ext == '.gif':
                        mime = 'image/gif'
                    else:
                        mime = 'application/octet-stream'

                    main_type, sub_type = mime.split('/', 1)
                    img_part = MIMEBase(main_type, sub_type)
                    img_part.set_payload(img_data)
                    encoders.encode_base64(img_part)
                    cid = f'image_{i}'
                    img_part.add_header('Content-ID', f'<{cid}>')
                    img_part.add_header('Content-Disposition', 'inline',
                                        filename=os.path.basename(img_path))
                    related.attach(img_part)
                except Exception as e:
                    print(f"[ERROR] Failed to attach inline image {img_path}: {e}")

        # Attach related (html + images) to outer message
        msg.attach(related)

        # Attach file attachments
        if attachments:
            for file_path in attachments:
                if not os.path.isfile(file_path):
                    print(f"[WARN] Attachment not found: {file_path}")
                    continue
                try:
                    with open(file_path, 'rb') as f:
                        mime_type, _ = mimetypes.guess_type(file_path)
                        if mime_type is None:
                            mime_type = 'application/octet-stream'
                        main_type, sub_type = mime_type.split('/', 1)
                        part = MIMEBase(main_type, sub_type)
                        part.set_payload(f.read())
                        encoders.encode_base64(part)
                        filename = os.path.basename(file_path)
                        part.add_header('Content-Disposition', 'attachment',
                                        filename=Header(filename, 'utf-8').encode())
                        msg.attach(part)
                except Exception as e:
                    print(f"[ERROR] Failed to attach file {file_path}: {e}")

        try:
            self.smtp_conn.send_message(msg)
            print(f"[OK] Email with images and attachments sent to {to}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to send email: {e}")
            traceback.print_exc()
            return False


    def logout(self):
        """关闭所有连接"""
        if self.imap_conn:
            try:
                self.imap_conn.logout()
                print("✅ IMAP已登出")
            except:
                pass
        if self.smtp_conn:
            try:
                self.smtp_conn.quit()
                print("✅ SMTP已断开")
            except:
                pass


# ========== 示例使用 ==========
if __name__ == "__main__":

    # 创建管理器实例
    manager = EmailManager()

    # print("\n=== 示例1：按关键字下载附件（xlsx） ===")
    # manager.download_attachments_by_keyword(["配邦中圣", "2026-03-30"], file_extensions=['.xlsx'], limit=10)


    # print("\n=== 示例2：按关键字下载图片 ===")
    # manager.download_images_by_keyword("照片", limit=5)
    #
    # print("\n=== 示例3：发送带附件的邮件 ===")
    # manager.send_email_with_attachments(
    #     to="recipient@example.com",
    #     subject="测试邮件",
    #     body="这是正文内容。",
    #     attachments=["E:\\code\\email\\20250320_120000_example.xlsx"]
    # )
    manager.send_email_with_attachments(['pagududeshengjiang@shpbjj.com'], '周度净值分析报告V1', '配邦恒升中性1号_20260330_20260403_分析报告', attachments=['/home/zhanggh/DailyScripts/PDF/配邦恒升中性1号_20260330_20260403_分析报告.pdf'])

    #
    # print("\n=== 示例4：发送内嵌图片邮件 ===")
    # # 注意：HTML中图片src必须与添加的图片cid对应，例如<img src="cid:image_0">
    # html = """
    # <html>
    #   <body>
    #     <h1>测试内嵌图片</h1>
    #     <img src="cid:image_0">
    #     <p>这是内嵌图片示例</p>
    #   </body>
    # </html>
    # """
    # manager.send_email_with_inline_images(
    #     to="recipient@example.com",
    #     subject="内嵌图片测试",
    #     body_html=html,
    #     image_paths=["E:\\code\\email\\test.jpg"]
    # )

    # 关闭连接
    manager.logout()