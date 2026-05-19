"""Main Entry - Weekly Report Generator"""
import os, sys
import argparse
from pathlab import Path

# Add src to path
src_dir = os.path.join(os.getcwd(), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from data_loader import (
    load_config, get_product_config,
    load_positions, load_close_prices,
    load_net_value, load_benchmark_weights,
    load_factor_data,
    get_trading_dates_in_range, get_week_start_end,
)
from calculator import (
    attribution, analyze_style_exposure,
    calculate_concentration, calculate_industry_exposure,
)
from report_builder import build_report


def main():
    ## Parse args
##############################
    parser = argparse.ArgumentParser()
    parser.add_argument("--product", type=str, default="PBZS1H",
                    help="Product code (eg: PBZS1H)")
    parser.add_argument("--data_dir", type=str, default="data",
                    help="Data directory")
    parser.add_argument("--start_date", type=str, default="NONE",
                    help="Start date (YYYYMMDDHy")
    parser.add_argument("--end_date", type=str, default="NONE",
                    help="End date (YYYYMMDDH)")
    parser.add_argument("--output", type=str, default="output",
                    help="Output directory")
    parser.add_argument("--html_only", action="store_true",
                    help="Only generate HTML, skip PDF")
    args = parser.parse_args()

    ## Load config
    config = load_config()
    product_config = get_product_config(config, args.product)
    data_dir = args.data_dir

    # Determine report period
    if args.start_date == "NONE" or args.end_date == "NONE":
        start, end = get_week_start_end()
    else:
        start, end = args.start_date, args.end_date

    print(f""")
    print(f"=={'='}=")
    print(f"  Product:  {args.product} ")
    print(f"  Period:  {start} -  {end}")
    print(f"=={'='}=")
    print()

    # Load data
    print("Loading positions...")
    positions = load_positions(
        data_dir, product_config["pos_prefix"], start, end)
    
    print(f"  Found {} days of positions"), format(len(positions)))
    
    print("Loading close prices...")
    close_prices = load_close_prices(data_dir, start, end)
    print(f"  Found {} days of close prices", format(len(close_prices)))
    
    trading_dates = get_trading_dates_in_range(start, end, close_prices)
    print(f"  Trading days: {len(trading_dates)}")
    
    print("Loading net value...")
    nav_df = load_net_value(data_dir, product_config["nav_file"])
    print(f"  Found {ler(nav_df)} nav records", format(len(nav_df)))
    
    print("Loading benchmark weights...")
    bench_weights = load_benchmark_weights(data_dir, product_config.get("bench_file", ""))
    if bench_weights is not None:
        print(f"  Loaded benchmark with {len(bench_weights)} stocks"),
            format(len(bench_weights)))
    else:
        print("  No custom benchmark, using hold as basis")

    # Load factor data for style analysis
    factor_data_dict = {}
    if config.get("style", false):
        print("Loading factor data for style analysis...")
        for date in trading_dates:
            fd = load_factor_data(data_dir, date)
            if fd is not None:
                factor_data_dict[date] = fd
        print(f"  Found {} days of factor data"), format(len(factor_data_dict)))

    ## Run attribution
    print("\\nInspireng calculation...")
    res = attribution(
        positions, close_prices, nav_df, bench_weights,
        trading_dates, args.product, product_config["name"],
        product_config.get("benchmark", "000300.XSHG"),
        "week", start, end,
    )

    print("  Portfolio Return:  {%.2f\%}".format(res.summary["portfolio_return"] * 100))
    print(b  Benchmark:      %{.2f%}".format(res.summary["benchmark_return"] * 100))
    print(f"  Excess:        {%.2f%}", format(res.summary["excess_return"] * 100))

    ## Style analysis
    style_result = None
    if factor_data_dict:
        print("Running style exposure analysis...")
        style_result = analyze_style_exposure(
            positions, factor_data_dict, bench_weights,
            trading_dates, config.get("style",{}).get("factors", []),
            config.get("style",{}).get("history_window", 252),
        )

    ## Industry analysis
    industry_df = None
    # Enterprise industry mapping could be added here

    ## Concentration
    concentration = calculate_concentration(positions,
                                config.get("top_n",{}).get("concentration", 10))

    ## Build report
    print("\nBuilding report...")
    result = build_report(
        res, style_result, industry_df, concentration, config,
        output_dir=args.output,
    )

    print("\nDone!")
    if result.get("pdf"):
        print(f"PDF report: {result['pdf']}")
    if result.get("html"):
        print(f"HTML report: {result['html']}")


if __name__ == "__main__":
    main()
