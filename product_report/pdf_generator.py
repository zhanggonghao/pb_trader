"""
PDF生成模块 - 生成专业的量化基金产品报告

本模块负责生成PDF格式的量化基金产品报告，包含：
- 净值信息图表
- 每日收益明细表格
- 个股权重Top10对比图
- 持仓集中度图表
- 因子风格敞口图表（双Y轴折线图）

所有配置参数均从config模块读取。
"""
import os
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import rcParams
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak, KeepTogether
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import config
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from io import BytesIO

matplotlib.use('Agg')

FONT_PATH_SIMHEI = r'C:\Windows\Fonts\simhei.ttf'
FONT_PATH_SIMSUN = r'C:\Windows\Fonts\simsunb.ttf'

if os.path.exists(FONT_PATH_SIMHEI):
    pdfmetrics.registerFont(TTFont('SimHei', FONT_PATH_SIMHEI))
if os.path.exists(FONT_PATH_SIMSUN):
    pdfmetrics.registerFont(TTFont('SimSun', FONT_PATH_SIMSUN))

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial']


def format_pct(value, decimals=2):
    """格式化百分比"""
    if value is None:
        return '-'
    return f"{value * 100:.{decimals}f}%"


def format_number(value, decimals=2):
    """格式化数字"""
    if value is None:
        return '-'
    return f"{value:,.{decimals}f}"


def format_money(value, decimals=2):
    """格式化金额"""
    if value is None:
        return '-'
    return f"¥{value:,.{decimals}f}"


def format_date_str(date_str):
    """格式化日期字符串为YYYYMMDD"""
    return str(date_str)


class PDFGenerator:
    def __init__(self, output_path):
        self.output_path = output_path
        self.doc = None
        self.story = []
        self.page_width = A4[0]
        self.page_height = A4[1]
        self.margin = 1.5 * cm

        self.primary_color = '#1a3a5c'
        self.secondary_color = '#2d5a87'
        self.accent_color = '#4a90d9'
        self.text_color = '#333333'
        self.light_gray = '#f5f5f5'
        self.border_color = '#cccccc'

        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=getSampleStyleSheet()['Title'],
            fontName='SimHei',
            fontSize=24,
            textColor=colors.HexColor(self.primary_color),
            spaceAfter=6,
            alignment=TA_CENTER,
        )

        self.subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=getSampleStyleSheet()['Normal'],
            fontName='SimHei',
            fontSize=14,
            textColor=colors.HexColor(self.secondary_color),
            spaceAfter=20,
            alignment=TA_CENTER,
        )

        self.section_style = ParagraphStyle(
            'Section',
            parent=getSampleStyleSheet()['Heading1'],
            fontName='SimHei',
            fontSize=14,
            textColor=colors.white,
            backColor=colors.HexColor(self.primary_color),
            spaceBefore=15,
            spaceAfter=10,
            leftIndent=-self.margin,
            rightIndent=-self.margin,
            borderPadding=(8, 10, 8, 10),
        )

        self.subsection_style = ParagraphStyle(
            'Subsection',
            parent=getSampleStyleSheet()['Heading2'],
            fontName='SimHei',
            fontSize=12,
            textColor=colors.HexColor(self.primary_color),
            spaceBefore=12,
            spaceAfter=6,
        )

        self.body_style = ParagraphStyle(
            'Body',
            parent=getSampleStyleSheet()['Normal'],
            fontName='SimHei',
            fontSize=10,
            textColor=colors.HexColor(self.text_color),
            spaceAfter=6,
        )

        self.table_header_style = ParagraphStyle(
            'TableHeader',
            parent=getSampleStyleSheet()['Normal'],
            fontName='SimHei',
            fontSize=9,
            textColor=colors.white,
            alignment=TA_CENTER,
        )

    def create_doc(self):
        """创建PDF文档"""
        self.doc = SimpleDocTemplate(
            self.output_path,
            pagesize=A4,
            rightMargin=self.margin,
            leftMargin=self.margin,
            topMargin=self.margin,
            bottomMargin=self.margin,
        )

    def add_title_page(self, product_name, period):
        """添加标题页"""
        self.story.append(Spacer(1, 3 * cm))

        title_text = f"<b>{product_name}</b>"
        self.story.append(Paragraph(title_text, self.title_style))

        subtitle_text = f"周期产品报告 | {period[0]} ~ {period[-1]}"
        self.story.append(Paragraph(subtitle_text, self.subtitle_style))

        self.story.append(Spacer(1, 2 * cm))

    def add_section_header(self, title):
        """添加章节标题"""
        header_table = Table(
            [[Paragraph(f"<b>{title}</b>", self.table_header_style)]],
            colWidths=[self.page_width - 2 * self.margin],
        )
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(self.primary_color)),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'SimHei'),
            ('FONTSIZE', (0, 0), (-1, -1), 13),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ]))
        self.story.append(header_table)
        self.story.append(Spacer(1, 0.3 * cm))

    def add_kpi_cards(self, kpis):
        """添加KPI指标卡片"""
        card_data = [[]]

        for i, (label, value, color) in enumerate(kpis):
            if i > 0 and i % 3 == 0:
                card_data.append([])
            card_data[-1].append((label, value, color))

        for row in card_data:
            row_data = []
            for label, value, color in row:
                cell_content = f'<para align="center"><font size="9" color="#666666">{label}</font><br/><font size="16" color="{color}"><b>{value}</b></font></para>'
                row_data.append(Paragraph(cell_content, self.body_style))

            if len(row_data) < 3:
                while len(row_data) < 3:
                    row_data.append('')

            card_table = Table([row_data], colWidths=[(self.page_width - 2 * self.margin) / 3] * 3)
            card_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(self.light_gray)),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor(self.border_color)),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor(self.border_color)),
                ('TOPPADDING', (0, 0), (-1, -1), 12),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ]))
            self.story.append(card_table)
            self.story.append(Spacer(1, 0.3 * cm))

    def create_nav_chart(self, calc_result, dates):
        """创建净值曲线图"""
        nav_curve = calc_result['nav_curve']
        bench_curve = calc_result['bench_curve']
        nav_dates = calc_result['nav_dates']

        nav_values = [nav_curve.get(d) for d in nav_dates]
        bench_values = [bench_curve.get(d) for d in nav_dates]

        norm_bench = []
        if bench_values[0] and bench_values[0] > 0:
            base_bench = bench_values[0]
            norm_bench = [v / base_bench * nav_values[0] if v else None for v in bench_values]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6),
                                        gridspec_kw={'height_ratios': [3, 1]})

        x_labels = [str(d) for d in nav_dates]
        x = range(len(nav_dates))

        if nav_values:
            ax1.plot(x, nav_values, 'b-', linewidth=2, marker='o', markersize=6, label='净值')
        if norm_bench:
            ax1.plot(x, norm_bench, 'r--', linewidth=1.5, marker='s', markersize=5, label='基准(归一化)')

        ax1.set_ylabel('净值', fontsize=11)
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(x)
        ax1.set_xticklabels(x_labels, fontsize=9)

        for i, (nav, bench) in enumerate(zip(nav_values, norm_bench)):
            if nav:
                ax1.annotate(f'{nav:.4f}', (i, nav), textcoords="offset points",
                            xytext=(0, 10), ha='center', fontsize=8, color='blue')

        # 正确的回撤计算：running_peak初始化为第一个净值，遍历时动态更新
        drawdowns = []
        if nav_values and nav_values[0]:
            running_peak = nav_values[0]  # 初始化为第一个净值
            for v in nav_values:
                if v:
                    if v > running_peak:
                        running_peak = v  # 更新历史峰值
                    dd = (running_peak - v) / running_peak if running_peak > 0 else 0
                    drawdowns.append(dd)
                else:
                    drawdowns.append(None)
        else:
            drawdowns = [0] * len(nav_values)

        ax2.fill_between(x, drawdowns, 0, alpha=0.3, color='gray', step='mid')
        ax2.plot(x, drawdowns, 'gray', linewidth=1)
        ax2.set_ylabel('回撤', fontsize=11)
        ax2.set_xlabel('日期', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks(x)
        ax2.set_xticklabels(x_labels, fontsize=9)
        ax2.set_ylim(bottom=0)

        plt.tight_layout()

        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        img_buffer.seek(0)
        return img_buffer

    def create_daily_return_table(self, module2_data, module3_data):
        """创建每日收益率表格"""
        headers = ['日期', '多头收益率', '基准收益率', '超额收益', '证券户贡献', '期货基差贡献', '合计贡献']

        data = [headers]
        for m2, m3 in zip(module2_data, module3_data):
            date_str = str(m2['date'])
            stk_ret = format_pct(m2['stk_return'])
            bench_ret = format_pct(m2['bench_return'])
            excess = format_pct(m2['excess'])
            stk_contrib = format_pct(m3['stk_contrib'])
            fut_contrib = format_pct(m3['fut_contrib'])
            total_contrib = format_pct(m3['total_contrib'])

            data.append([date_str, stk_ret, bench_ret, excess, stk_contrib, fut_contrib, total_contrib])

        col_widths = [(self.page_width - 2 * self.margin) / 7] * 7

        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.primary_color)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'SimHei'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTNAME', (0, 1), (-1, -1), 'SimHei'),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor(self.border_color)),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor(self.light_gray)]),
        ]))

        return table

    def create_account_detail_table(self, module3_data, accounts):
        """创建账户详细信息表格"""
        all_data = []

        for m3 in module3_data:
            date_str = str(m3['date'])

            for acc in m3['accounts']:
                acc_name = acc['account']
                ret = format_pct(acc['return'])
                excess = format_pct(acc['excess'])
                pnl = format_money(acc['pnl'])
                net = format_money(acc['net_assets'])

                all_data.append([date_str, acc_name, ret, excess, pnl, net])

            all_data.append(['', '', '', '', '', ''])

        if not all_data or len(all_data) == 0:
            return None

        headers = ['日期', '账户', '收益率', '超额', '收益(¥)', '净资产(¥)']
        data = [headers] + all_data

        col_widths = [(self.page_width - 2 * self.margin) / 6] * 6

        table = Table(data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.primary_color)),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'SimHei'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTNAME', (0, 1), (-1, -1), 'SimHei'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor(self.border_color)),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor(self.light_gray)]),
        ]))

        return table

    def create_top10_table(self, module4_data):
        """创建Top10股权重表格"""
        all_tables = []

        for date_str, data in module4_data.items():
            headers = ['序号', '股票代码', '持股数', '市值(¥)', '权重']

            rows = []
            for i, item in enumerate(data['top10']):
                code = str(item['code'])
                rows.append([
                    str(i + 1),
                    code,
                    format_number(item['hold'], 0),
                    format_money(item['market_value']),
                    format_pct(item['weight']),
                ])

            table_data = [headers] + rows
            col_widths = [
                0.8 * cm,
                2.5 * cm,
                2 * cm,
                3 * cm,
                2 * cm,
            ]

            table = Table(table_data, colWidths=col_widths)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.secondary_color)),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'SimHei'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTNAME', (0, 1), (-1, -1), 'SimHei'),
                ('FONTSIZE', (0, 1), (-1, -1), 8),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor(self.border_color)),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor(self.light_gray)]),
            ]))

            title = Paragraph(f"<b>{date_str} Top10持仓</b>", self.subsection_style)
            all_tables.append((title, table))

        return all_tables

    def create_combined_top10_chart(self, module4_data):
        """创建组合的Top10股权重图表（5天合并到一张图）"""
        dates = sorted(module4_data.keys())
        n_dates = len(dates)

        if n_dates == 0:
            return None

        fig, ax = plt.subplots(figsize=(14, 6))

        bar_width = 0.08
        x_base = np.arange(10)

        colors_list = plt.cm.Set3(np.linspace(0, 1, n_dates))

        for i, date_str in enumerate(dates):
            data = module4_data[date_str]
            weights = [item['weight'] * 100 for item in data['top10']]
            x_pos = x_base + i * bar_width
            ax.bar(x_pos, weights, bar_width, label=date_str, color=colors_list[i])

        ax.set_xlabel('排名', fontsize=11)
        ax.set_ylabel('权重 (%)', fontsize=11)
        ax.set_title('每日个股权重Top10对比', fontsize=13, fontweight='bold')
        ax.set_xticks(x_base + bar_width * (n_dates - 1) / 2)
        ax.set_xticklabels([str(j + 1) for j in range(10)], fontsize=10)
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        img_buffer.seek(0)
        return img_buffer

    def create_concentration_chart(self, module5_data):
        """创建持仓集中度图表（5天合并到一张图）"""
        dates = sorted(module5_data.keys())

        if len(dates) == 0:
            return []

        fig, ax = plt.subplots(figsize=(12, 5))

        x = np.arange(len(dates))
        bar_width = 0.6

        top_20_weights = [module5_data[d]['top_20_pct_weight'] * 100 for d in dates]

        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(dates)))

        bars = ax.bar(x, top_20_weights, bar_width, color=colors, edgecolor='white', linewidth=1)

        for i, (bar, val) in enumerate(zip(bars, top_20_weights)):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                   f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(dates, fontsize=10)
        ax.set_ylabel('累计权重 (%)', fontsize=11)
        ax.set_title('持仓集中度（前20%股票市值占比）', fontsize=13, fontweight='bold')
        ax.set_ylim(0, max(top_20_weights) * 1.15)
        ax.grid(axis='y', alpha=0.3)

        for i, d in enumerate(dates):
            info = module5_data[d]
            ax.text(i, -5, f"({info['top_20_pct_count']}/{info['total_stocks']})",
                   ha='center', va='top', fontsize=8, color='gray')

        plt.tight_layout()

        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        img_buffer.seek(0)
        return [('combined', img_buffer)]

    def create_factor_chart(self, module6_data):
        """创建因子风格敞口图表（按因子分组，每个因子一张图）"""
        all_charts = []

        # 使用新的因子配置中的因子名称映射
        factors_cn = config.FactorConfig.SELECTED_FACTORS

        dates = sorted(module6_data.keys())
        if not dates:
            return []

        for factor_en, factor_cn in factors_cn.items():
            portfolio_vals = []
            benchmark_vals = []
            
            for d in dates:
                data = module6_data[d]
                portfolio = data.get('portfolio', {})
                benchmark = data.get('benchmark', {})
                portfolio_vals.append(portfolio.get(factor_en, 0))
                benchmark_vals.append(benchmark.get(factor_en, 0))

            fig, ax = plt.subplots(figsize=(10, 4))

            x = np.arange(len(dates))

            # 使用单个纵坐标轴绘制组合和基准
            ax.plot(x, portfolio_vals, marker='o', label='组合', color=self.accent_color, linewidth=2)
            ax.plot(x, benchmark_vals, marker='s', label='基准', color='#e74c3c', linewidth=2)
            
            ax.set_xlabel('日期', fontsize=10)
            ax.set_ylabel('因子暴露', fontsize=10)
            ax.set_title(f'{factor_cn} 因子敞口', fontsize=11, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(dates, fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.legend(loc='upper right', fontsize=9)

            plt.tight_layout()

            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()

            img_buffer.seek(0)
            all_charts.append((factor_cn, img_buffer))

        return all_charts

    def generate_market_review_text(self, market_review_data, calc_result):
        """生成市场回顾文字段落"""
        if not market_review_data:
            return ""
        
        # 格式化百分比
        def fmt_pct(val):
            if val is None:
                return "--"
            return f"{val * 100:.2f}%"
        
        # 格式化数字（指数点位）
        def fmt_num(val):
            if val is None:
                return "--"
            return f"{val:.2f}"
        
        # 获取数据
        sh_return = market_review_data.get('sh_index_return')
        hs300_return = market_review_data.get('hs300_return')
        zz500_return = market_review_data.get('zz500_return')
        zz1000_return = market_review_data.get('zz1000_return')
        cyb_return = market_review_data.get('cyb_return')
        
        sh_close = market_review_data.get('sh_index_close')
        hs300_close = market_review_data.get('hs300_close')
        
        if_return = market_review_data.get('if_return')
        ic_return = market_review_data.get('ic_return')
        im_return = market_review_data.get('im_return')
        
        basis = market_review_data.get('basis')
        basis_pct = market_review_data.get('basis_pct')
        basis_far = market_review_data.get('basis_far')
        basis_change = market_review_data.get('basis_change')
        basis_far_change = market_review_data.get('basis_far_change')
        
        avg_turnover = market_review_data.get('avg_turnover')
        
        stk_contrib = market_review_data.get('total_stk_contrib')
        fut_contrib = market_review_data.get('total_fut_contrib')
        
        # 产品数据
        nav_growth = calc_result.get('nav_growth')
        max_dd = calc_result.get('max_drawdown')
        
        # 辅助函数：涨跌描述
        def up_down(val):
            if val is None:
                return ""
            return "周期涨幅" if val >= 0 else "周期跌幅"
        
        # 计算超额收益和基差收益的具体数值（用于对比描述）
        total_contrib = (stk_contrib if stk_contrib else 0) + (fut_contrib if fut_contrib else 0)
        
        # 生成超额收益和基差收益的对比描述
        def generate_comparison_text(contrib, name):
            """生成某一项收益与基准的对比描述"""
            if contrib is None:
                return ""
            if contrib > 0:
                return f"{name}为{contrib*100:.2f}%，"
            elif contrib < 0:
                return f"{name}为{contrib*100:.2f}%；"
            else:
                return f"{name}为0.00%，与基准持平；"
        
        # 生成收益归因描述（根据股票端和基差端的正负组合及总贡献）
        def generate_contribution_text(stk, fut, total):
            """根据股票端和基差端的正负情况及总贡献生成描述"""
            if stk is None or fut is None:
                return ""
            
            stk_positive = stk > 0
            fut_positive = fut > 0
            total_positive = total > 0 if total else False
            
            # 四种情况的完整覆盖
            if stk_positive and fut_positive:
                # 股票正，基差正
                return "股票端超额贡献为正，基差端亦形成正向补充，共同推动产品净值上涨。"
            elif stk_positive and not fut_positive:
                # 股票正，基差负
                if total_positive:
                    return "股票端超额贡献为正，虽基差端贡献为负，但两者合计仍为正贡献，共同推动产品净值上涨。"
                else:
                    return "股票端超额贡献为正，但基差端贡献为负，两者合计为负贡献，未能完全形成正向补充。"
            elif not stk_positive and fut_positive:
                # 股票负，基差正
                if total_positive:
                    return "基差端贡献为正且完全对冲股票端超额亏损，两者合计为正贡献，共同推动产品净值上涨。"
                else:
                    return "基差端贡献为正但不足以弥补股票端超额亏损，两者合计为负贡献，共同拖累产品净值表现。"
            else:
                # 股票负，基差负
                return "股票端超额亏损与基差端负贡献共同拖累产品净值表现。"
        
        comparison_text = generate_comparison_text(stk_contrib, "超额收益") + generate_comparison_text(fut_contrib, "基差收益")
        contribution_text = generate_contribution_text(stk_contrib, fut_contrib, total_contrib)
        
        # 生成文字
        text = f"<b>市场回顾</b><br/><br/>" \
               f"周期内A股市场整体震荡{'走高' if (sh_return and sh_return >= 0) else '走低'}，大盘指数录得{'上涨' if (sh_return and sh_return >= 0) else '下跌'}。" \
               f"上证指数收于{fmt_num(sh_close)}点，{up_down(sh_return)}{fmt_pct(sh_return)}；" \
               f"沪深300指数收于{fmt_num(hs300_close)}点，{up_down(hs300_return)}{fmt_pct(hs300_return)}。" \
               f"风格方面，大盘蓝筹{'表现优于' if (hs300_return and zz500_return and hs300_return > zz500_return) else '表现弱于'}中小盘，" \
               f"中证500{up_down(zz500_return)}{fmt_pct(zz500_return)}，中证1000{up_down(zz1000_return)}{fmt_pct(zz1000_return)}，" \
               f"创业板指{up_down(cyb_return)}{fmt_pct(cyb_return)}，市场呈现一定的风格分化特征。" \
               f"两市日均成交额约{fmt_num(avg_turnover)}万亿元，显示市场交投情绪{'边际回暖' if (avg_turnover and avg_turnover > 1.0) else '相对平淡'}。<br/><br/>" \
               f"股指期货方面，IF当月合约{up_down(if_return)}{fmt_pct(if_return)}，" \
               f"基差{'收敛' if (basis_change and basis_change < 0) else '拉开' if (basis_change and basis_change > 0) else '变化不大'}；" \
               f"最远月合约（IF2612）基差{'收敛' if (basis_far_change and basis_far_change < 0) else '拉开' if (basis_far_change and basis_far_change > 0) else '变化不大'}。" \
               f"IC及IM合约分别{up_down(ic_return)}{fmt_pct(ic_return)}和{up_down(im_return)}{fmt_pct(im_return)}，中小盘期货贴水幅度未见显著扩大。<br/><br/>" \
               f"产品方面，{config.REPORT_PRODUCT_NAME}周期内净值{up_down(nav_growth)}{fmt_pct(nav_growth)}，" \
               f"最大回撤为{fmt_pct(max_dd)}。" \
               f"从收益归因来看，周期内股票端超额贡献约{fmt_pct(stk_contrib)}，" \
               f"期货基差端贡献约{fmt_pct(fut_contrib)}。{contribution_text}"
        
        return text

    def generate_module1(self, calc_result, market_review_data=None):
        """生成模块1"""
        self.add_section_header("一、周期净值信息")

        kpis = [
            ('净值增长', format_pct(calc_result['nav_growth']), '#1a3a5c' if calc_result['nav_growth'] and calc_result['nav_growth'] >= 0 else '#c0392b'),
            ('基准收益率', format_pct(calc_result['bench_growth']), '#27ae60'),
            ('净值回撤', format_pct(calc_result['max_drawdown']), '#c0392b'),
        ]
        self.add_kpi_cards(kpis)

        # 添加市场回顾文字
        if market_review_data:
            review_text = self.generate_market_review_text(market_review_data, calc_result)
            if review_text:
                self.story.append(Spacer(1, 0.4 * cm))
                review_para = Paragraph(review_text, self.body_style)
                self.story.append(review_para)
                self.story.append(Spacer(1, 0.4 * cm))

        chart_img = self.create_nav_chart(calc_result, calc_result['nav_dates'])
        img = Image(chart_img, width=(self.page_width - 2 * self.margin), height=(self.page_width - 2 * self.margin) * 0.6)
        self.story.append(Spacer(1, 0.3 * cm))
        self.story.append(img)

        self.story.append(Spacer(1, 0.5 * cm))

    def generate_module2(self, module2_data, module3_data):
        """生成模块2"""
        self.add_section_header("二、每日收益明细")

        self.story.append(Paragraph("<b>每日收益率及贡献分解</b>", self.subsection_style))

        table = self.create_daily_return_table(module2_data, module3_data)
        self.story.append(table)

        self.story.append(Spacer(1, 0.5 * cm))

    def generate_module3(self, module3_data, accounts):
        """生成模块3"""
        self.add_section_header("三、多空详情")

        for m3 in module3_data:
            date_str = str(m3['date'])

            detail_text = f"<b>{date_str} 各账户明细</b><br/>"
            detail_text += f"产品总净资产: {format_money(m3['product_assets'])} | "
            detail_text += f"证券户合计收益率: {format_pct(m3['total_stk_return'])} | "
            detail_text += f"期货对冲收益率: {format_pct(m3['fut_return'])} | "
            detail_text += f"基差收益率: {format_pct(m3['basis_return'])}<br/>"
            detail_text += f"证券户超额贡献: {format_pct(m3['stk_contrib'])} | "
            detail_text += f"期货基差贡献: {format_pct(m3['fut_contrib'])} | "
            detail_text += f"合计贡献: {format_pct(m3['total_contrib'])}"

            self.story.append(Paragraph(detail_text, self.body_style))

            acc_table_data = [['账户', '收益率', '超额', '收益', '净资产']]
            for acc in m3['accounts']:
                acc_table_data.append([
                    acc['account'],
                    format_pct(acc['return']),
                    format_pct(acc['excess']),
                    format_money(acc['pnl']),
                    format_money(acc['net_assets']),
                ])

            col_widths = [(self.page_width - 2 * self.margin) / 5] * 5
            acc_table = Table(acc_table_data, colWidths=col_widths)
            acc_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(self.secondary_color)),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'SimHei'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('FONTNAME', (0, 1), (-1, -1), 'SimHei'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor(self.border_color)),
                ('TOPPADDING', (0, 0), (-1, -1), 5),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor(self.light_gray)]),
            ]))

            self.story.append(acc_table)
            self.story.append(Spacer(1, 0.3 * cm))

        self.story.append(PageBreak())

    def generate_module4(self, module4_data):
        """生成模块4"""
        self.add_section_header("四、每日个股权重Top10")

        chart_img = self.create_combined_top10_chart(module4_data)
        if chart_img:
            img = Image(chart_img, width=(self.page_width - 2 * self.margin), height=(self.page_width - 2 * self.margin) * 0.45)
            self.story.append(img)

        self.story.append(Spacer(1, 0.5 * cm))

        self.story.append(PageBreak())

    def generate_module5(self, module5_data):
        """生成模块5"""
        self.add_section_header("五、持仓集中度")

        charts = self.create_concentration_chart(module5_data)

        for label, chart_img in charts:
            img = Image(chart_img, width=(self.page_width - 2 * self.margin), height=(self.page_width - 2 * self.margin) * 0.4)
            self.story.append(img)

        self.story.append(Spacer(1, 0.3 * cm))

        self.story.append(PageBreak())

    def generate_module6(self, module6_data):
        """生成模块6"""
        self.add_section_header("七、因子风格敞口")

        charts = self.create_factor_chart(module6_data)

        for i, (factor_cn, chart_img) in enumerate(charts):
            img = Image(chart_img, width=(self.page_width - 2 * self.margin), height=(self.page_width - 2 * self.margin) * 0.35)
            self.story.append(Paragraph(f"<b>{factor_cn}</b>", self.subsection_style))
            self.story.append(img)
            if (i + 1) % 3 == 0:
                self.story.append(Spacer(1, 0.3 * cm))
            else:
                self.story.append(Spacer(1, 0.2 * cm))

        if not charts:
            self.story.append(Paragraph("暂无因子数据", self.body_style))

        self.story.append(PageBreak())

    def create_industry_comparison_chart(self, module7_data):
        """创建行业对比图表（组合 vs 基准，按日期分图）"""
        all_charts = []

        dates = sorted(module7_data.keys())
        if not dates:
            return []

        for d in dates:
            data = module7_data[d]
            portfolio_ind = data.get('portfolio_industry', {})
            benchmark_ind = data.get('benchmark_industry', {})

            if not portfolio_ind:
                continue

            # 合并所有行业
            all_industries = sorted(set(list(portfolio_ind.keys()) + list(benchmark_ind.keys())))

            if not all_industries:
                continue

            portfolio_vals = [portfolio_ind.get(ind, 0) * 100 for ind in all_industries]
            benchmark_vals = [benchmark_ind.get(ind, 0) * 100 for ind in all_industries]
            diff_vals = [p - b for p, b in zip(portfolio_vals, benchmark_vals)]

            # 行业名称直接作为标签
            labels = all_industries

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                            gridspec_kw={'height_ratios': [3, 2]})

            x = np.arange(len(all_industries))
            bar_width = 0.35

            # 上图：组合 vs 基准行业权重对比
            bars1 = ax1.bar(x - bar_width / 2, portfolio_vals, bar_width,
                           label='组合', color=self.accent_color, alpha=0.85)
            bars2 = ax1.bar(x + bar_width / 2, benchmark_vals, bar_width,
                           label='基准(沪深300)', color='#e74c3c', alpha=0.85)

            ax1.set_ylabel('权重 (%)', fontsize=11)
            ax1.set_title(f'{d} 行业分布对比（申万一级行业）', fontsize=13, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
            ax1.legend(fontsize=10, loc='upper right')
            ax1.grid(axis='y', alpha=0.3)

            # 下图：行业偏离度
            colors_diff = ['#27ae60' if v >= 0 else '#c0392b' for v in diff_vals]
            ax2.bar(x, diff_vals, bar_width * 1.2, color=colors_diff, alpha=0.8)
            ax2.axhline(y=0, color='black', linewidth=0.5)
            ax2.set_ylabel('偏离度 (%)', fontsize=11)
            ax2.set_xlabel('行业', fontsize=11)
            ax2.set_title('行业偏离度（组合 - 基准）', fontsize=11, fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
            ax2.grid(axis='y', alpha=0.3)

            plt.tight_layout()

            img_buffer = BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()

            img_buffer.seek(0)
            all_charts.append((d, img_buffer))

        return all_charts

    def generate_module7(self, module7_data):
        """生成模块7：行业对比"""
        self.add_section_header("六、行业分布对比")

        charts = self.create_industry_comparison_chart(module7_data)

        for date_str, chart_img in charts:
            img = Image(chart_img, width=(self.page_width - 2 * self.margin),
                      height=(self.page_width - 2 * self.margin) * 0.55)
            self.story.append(img)
            self.story.append(Spacer(1, 0.3 * cm))

        if not charts:
            self.story.append(Paragraph("暂无行业数据", self.body_style))

        self.story.append(PageBreak())

    def build(self, calc_result, module2_data, module3_data, module4_data, module5_data, module6_data, module7_data, dates, market_review_data=None):
        """构建完整的PDF报告"""
        self.create_doc()

        period_str = f"{dates[0]} ~ {dates[-1]}"
        self.add_title_page(config.REPORT_PRODUCT_NAME, dates)
        self.story.append(PageBreak())
        # print(self.story)

        self.generate_module1(calc_result, market_review_data)
        self.generate_module2(module2_data, module3_data)
        # 暂时跳过模块3 - 多空详情
        # self.generate_module3(module3_data, dates)
        self.generate_module4(module4_data)
        self.generate_module5(module5_data)
        self.generate_module7(module7_data)
        self.generate_module6(module6_data)

        self.doc.build(self.story)
        print(f"PDF报告已生成: {self.output_path}")
