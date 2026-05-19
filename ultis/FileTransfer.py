import paramiko
import os
import stat
from scp import SCPClient, SCPException


class LinuxFileTransfer:
    def __init__(self, hostname="192.168.1.168", username="zhanggh", password="zhanggh@123.com", port=51622):
        self.hostname = hostname
        self.username = username
        self.password = password
        self.port = port
        self.ssh_client = None

    def connect(self):
        """建立SSH连接"""
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            self.ssh_client.connect(
                hostname=self.hostname,
                port=self.port,
                username=self.username,
                password=self.password
            )
            print("SSH连接成功")
        except Exception as e:
            print(f"SSH连接失败: {e}")
            self.ssh_client = None

    def disconnect(self):
        """关闭SSH连接"""
        if self.ssh_client:
            self.ssh_client.close()
            print("SSH连接已断开")

    # ==================== 单文件传输 ====================

    def upload_file(self, local_path, remote_path):
        """上传单个文件到Linux服务器"""
        if not self.ssh_client:
            print("请先调用 connect() 建立连接")
            return

        try:
            with SCPClient(self.ssh_client.get_transport()) as scp:
                scp.put(local_path, remote_path)
                print(f"文件 {local_path} 已上传到 {remote_path}")
        except SCPException as e:
            print(f"文件上传失败: {e}")

    def download_file(self, remote_path, local_path):
        """从Linux服务器下载单个文件"""
        if not self.ssh_client:
            print("请先调用 connect() 建立连接")
            return

        try:
            with SCPClient(self.ssh_client.get_transport()) as scp:
                scp.get(remote_path, local_path)
                print(f"文件 {remote_path} 已下载到 {local_path}")
        except SCPException as e:
            print(f"文件下载失败: {e}")

    # ==================== 文件夹传输（递归） ====================

    def download_folder(self, remote_dir, local_dir):
        """
        递归下载远程文件夹到本地指定文件夹
        :param remote_dir: 远程目录路径，如 "/home/trading/data"
        :param local_dir:  本地目录路径，如 r"G:\backup\data"
        """
        if not self.ssh_client:
            print("请先调用 connect() 建立连接")
            return

        # 确保本地目标目录存在
        os.makedirs(local_dir, exist_ok=True)

        try:
            sftp = self.ssh_client.open_sftp()
            self._download_folder_recursive(sftp, remote_dir, local_dir)
            sftp.close()
            print(f"文件夹下载完成: {remote_dir} -> {local_dir}")
        except Exception as e:
            print(f"文件夹下载失败: {e}")

    def _download_folder_recursive(self, sftp, remote_dir, local_dir):
        """递归下载辅助方法"""
        for item in sftp.listdir_attr(remote_dir):
            remote_path = f"{remote_dir}/{item.filename}".replace("//", "/")
            local_path = os.path.join(local_dir, item.filename)

            if stat.S_ISDIR(item.st_mode):
                # 是目录：创建本地目录，递归进入
                os.makedirs(local_path, exist_ok=True)
                print(f"  进入目录: {remote_path}")
                self._download_folder_recursive(sftp, remote_path, local_path)
            else:
                # 是文件：下载
                sftp.get(remote_path, local_path)
                print(f"  下载: {remote_path} -> {local_path}")

    def upload_folder(self, local_dir, remote_dir):
        """
        递归上传本地文件夹到远程指定文件夹
        :param local_dir:  本地目录路径，如 r"G:\data"
        :param remote_dir: 远程目录路径，如 "/home/trading/data"
        """
        if not self.ssh_client:
            print("请先调用 connect() 建立连接")
            return

        if not os.path.isdir(local_dir):
            print(f"本地文件夹不存在: {local_dir}")
            return

        # 确保远程目标目录存在
        self.mkdir_remote_folder(remote_dir)

        try:
            sftp = self.ssh_client.open_sftp()
            self._upload_folder_recursive(sftp, local_dir, remote_dir)
            sftp.close()
            print(f"文件夹上传完成: {local_dir} -> {remote_dir}")
        except Exception as e:
            print(f"文件夹上传失败: {e}")

    def _upload_folder_recursive(self, sftp, local_dir, remote_dir):
        """递归上传辅助方法"""
        for item in os.listdir(local_dir):
            local_path = os.path.join(local_dir, item)
            remote_path = f"{remote_dir}/{item}".replace("//", "/")

            if os.path.isdir(local_path):
                # 是目录：创建远程目录，递归进入
                self.mkdir_remote_folder(remote_path)
                print(f"  创建/进入目录: {remote_path}")
                self._upload_folder_recursive(sftp, local_path, remote_path)
            else:
                # 是文件：上传
                sftp.put(local_path, remote_path)
                print(f"  上传: {local_path} -> {remote_path}")

    # ==================== 远程目录操作 ====================

    def list_remote_dir(self, remote_path):
        """列出远程目录内容"""
        if not self.ssh_client:
            print("请先调用 connect() 建立连接")
            return []

        try:
            sftp = self.ssh_client.open_sftp()
            items = sftp.listdir(remote_path)
            sftp.close()
            return items
        except Exception as e:
            print(f"列出远程目录失败: {e}")
            return []

    def download_remote_files(self, remote_path, local_path):
        """
        下载远程目录下的所有文件（非递归，仅一级）
        注意：此方法不关闭连接，调用者自行管理
        """
        if not self.ssh_client:
            print("请先调用 connect() 建立连接")
            return

        os.makedirs(local_path, exist_ok=True)

        try:
            sftp = self.ssh_client.open_sftp()
            files = sftp.listdir(remote_path)
            print(f"远程目录 {remote_path} 包含: {files}")
            for filename in files:
                remote_file = f"{remote_path}/{filename}".replace("//", "/")
                local_file = os.path.join(local_path, filename)
                try:
                    sftp.get(remote_file, local_file)
                    print(f"  下载: {remote_file} -> {local_file}")
                except Exception as e:
                    print(f"  跳过 {filename}: {e}")
            sftp.close()
        except Exception as e:
            print(f"列出远程目录失败: {e}")

    def mkdir_remote_folder_sftp(self, remote_path):
        """通过SFTP创建远程目录"""
        if not self.ssh_client:
            print("请先调用 connect() 建立连接")
            return

        try:
            sftp = self.ssh_client.open_sftp()
            try:
                sftp.chdir(remote_path)
                print(f"远程目录已存在: {remote_path}")
            except IOError:
                sftp.mkdir(remote_path)
                print(f"远程目录创建成功: {remote_path}")
            sftp.close()
        except Exception as e:
            print(f"SFTP操作失败: {e}")

    def mkdir_remote_folder(self, remote_path):
        """通过shell命令创建远程目录（支持递归创建）"""
        if not self.ssh_client:
            print("请先调用 connect() 建立连接")
            return

        stdin, stdout, stderr = self.ssh_client.exec_command(
            f'test -d "{remote_path}" && echo "exists" || echo "not exists"'
        )
        folder_status = stdout.read().decode().strip()

        if folder_status == "not exists":
            print(f"远程目录不存在，正在创建: {remote_path}")
            stdin, stdout, stderr = self.ssh_client.exec_command(f'mkdir -p "{remote_path}"')
            error = stderr.read().decode()
            if error:
                print(f"创建失败: {error}")
            else:
                print(f"远程目录创建成功: {remote_path}")
        else:
            print(f"远程目录已存在: {remote_path}")


if __name__ == "__main__":
    # 服务器连接信息
    linux_server = "192.168.1.168"
    port = 51622
    user = "xtrader"
    pwd = "xtrader@123.com"

    transfer = LinuxFileTransfer(hostname=linux_server, username=user, password=pwd, port=port)
    transfer.connect()

    # ========== 单文件传输示例 ==========
    # transfer.upload_file(r"G:\test.txt", "/home/trading/test.txt")
    # transfer.download_file("/home/trading/test.txt", r"G:\test.txt")

    # ========== 文件夹传输示例 ==========
    # transfer.download_folder("/home/trading/data", r"G:\backup\data")
    # transfer.upload_folder(r"G:\data", "/home/trading/data")

    # ========== 列出远程目录 ==========
    # items = transfer.list_remote_dir("/home/trading")
    # print(items)

    transfer.disconnect()
