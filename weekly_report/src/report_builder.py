"""Report Builder - HTML generation + PDF via weasyprint"""
import os, shutil
from datetime import datetime
import jinja2
import pandas as pd
from calculator import AttributionResult, StyleExposureResult
import charts

def build_report(res, style_result, industry_df, concentration, config, output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "output")
    product_code = res.product_code
    end_date = res.end_date
    out_dir = os.path.join(output_dir, "reports", f"{product_code}_{end_date}")
    chart_dir = os.path.join(out_dir, "charts")
    os.makedirs(chart_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    top_n_config = config.get("top_n", {})
    stock_top_n = top_n_config.get("contribution_stocks", 10)
    conc_top_n = top_n_config.get("concentration", 10)

    print("Generating charts...")
    charts.plot_netvalue_curves(res, os.path.join(chart_dir, "netvalue.png"))
    charts.plot_daily_attribution(res, os.path.join(chart_dir, "daily_attr.png"))
    charts.plot_stock_contribution(res, os.path.join(chart_dir, "stock_contrib.png"), top_n=stock_top_n)

    has_style = style_result is not None
    if has_style:
        print("Generating style charts...")
        charts.plot_style_radar(style_result, os.path.join(chart_dir, "style_radar.png"))
        charts.plot_style_bars(style_result, os.path.join(chart_dir, "style_bars.png"))
        charts.plot_style_timeseries(style_result, os.path.join(chart_dir, "style_timeseries.png"))

    if industry_df is not None and not industry_df.empty:
        print("Generating industry chart...")
        charts.plot_industry_distribution(industry_df, os.path.join(chart_dir, "industry.png"))

    if concentration:
        print("Generating concentration chart...")
        charts.plot_concentration(concentration, os.path.join(chart_dir, "concentration.png"), top_n=conc_top_n)

    daily_rows = []
    for _, row in res.daily.iterrows():
        r = row.to_dict()
        if hasattr(r["date"], "strftime"):
            r["date"] = r["date"].strftime("%Y-%m-%d")
        else:
            r["date"] = str(r["date"])
        daily_rows.append(r)

    style_rows = []
    if has_style:
        for f, label in zip(style_result.factors, style_result.factor_labels):
            pe = style_result.portfolio_exposure
            be = style_result.benchmark_exposure
            ae = style_result.active_exposure
            act = pe.iloc[0].get(f) if f in pe.columns else None
            bench = be.iloc[0].get(f) if f in be.columns else None
            gap = ae.iloc[0].get(f) if f in ae.columns else None
            z = float(style_result.z_scores.get(f, 0)) if hasattr(style_result.z_scores, "get") else 0
            style_rows.append({"label": label, "actual": act, "benchmark": bench, "gap": gap, "z_score": z})

    template_dir = os.path.join(os.getcwd(), "templates")
    css_src = os.path.join(template_dir, "styles.css")
    if os.path.exists(css_src):
        shutil.copy(css_src, os.path.join(chart_dir, "styles.css"))

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(template_dir)))
    template = env.get_template("weekly_report.html")

    html = template.render(
        product_name=res.product_name,
        benchmark=res.benchmark,
        start_date=res.start_date,
        end_date=res.end_date,
        summary=res.summary,
        daily_rows=daily_rows,
        style_rows=style_rows,
        has_style=has_style,
        top_n=stock_top_n,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    html_path = os.path.join(out_dir, "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML report: {html_path}")

    pdf_path = None
    try:
        import weasyprint
        pdf_path = os.path.join(out_dir, "report.pdf")
        print("Converting to PDF...")
        weasyprint.HTML(string=html, base_url=str(chart_dir)).write_pdf(str(pdf_path))
        print(f"PDF report: {pdf_path}")
    except ImportError:
        print("Warning: weasyprint not installed, skip PDF")
    except Exception as e:
        print(f"Warning: PDF failed: {e}")

    print(f"Report saved in: {out_dir}")
    return {"html": html_path, "pdf": pdf_path}
