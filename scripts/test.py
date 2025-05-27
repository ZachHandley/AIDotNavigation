import asyncio
import json
from typing import Any, Dict, List, Optional
from typer import Typer, Option
from app.object_agent import ObjectAgent, ObjectAnalysisParams
from app.config import get_settings
from llama_index.core.workflow import StartEvent

app = Typer()

def create_large_ecommerce_data():
    """Generate large e-commerce dataset for testing"""
    return {
        "e_commerce_data": {
            "customers": [
                {
                    "id": i,
                    "name": f"Customer {i}",
                    "email": f"customer{i}@example.com",
                    "segment": (
                        "high_value"
                        if i <= 25
                        else "regular" if i <= 75 else "occasional"
                    ),
                    "registration_date": f"2023-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                    "orders": [
                        {
                            "order_id": f"order_{i}_{j}",
                            "date": f"2024-{(j % 12) + 1:02d}-{((i + j) % 28) + 1:02d}",
                            "status": "completed",
                            "items": [
                                {
                                    "product_id": f"product_{k}",
                                    "product": f"Product {k}",
                                    "category": [
                                        "Electronics",
                                        "Clothing",
                                        "Books",
                                        "Home",
                                    ][k % 4],
                                    "price": k * 10 + (i % 5) * 5,
                                    "quantity": k + (i % 3),
                                    "discount": 0.1 if i % 5 == 0 else 0,
                                }
                                for k in range(1, 4 + (i % 3))
                            ],
                            "shipping_cost": 10 if i % 3 == 0 else 0,
                            "total": sum(
                                (k * 10 + (i % 5) * 5)
                                * (k + (i % 3))
                                * (0.9 if i % 5 == 0 else 1.0)
                                for k in range(1, 4 + (i % 3))
                            )
                            + (10 if i % 3 == 0 else 0),
                        }
                        for j in range(
                            1, 3 + (i % 4)
                        )  # Variable number of orders per customer
                    ],
                }
                for i in range(1, 150)  # More customers
            ],
            "products": {
                f"product_{i}": {
                    "name": f"Product {i}",
                    "category": ["Electronics", "Clothing", "Books", "Home"][i % 4],
                    "price": i * 15 + (i % 10) * 3,
                    "cost": (i * 15 + (i % 10) * 3) * 0.6,  # 60% cost ratio
                    "stock": 100 + (i % 50),
                    "reviews": [
                        {
                            "customer_id": f"customer_{(j * 3 + i) % 149 + 1}",
                            "rating": 3 + (i + j) % 3,
                            "comment": f"Review {j} for product {i}",
                            "date": f"2024-{(j % 12) + 1:02d}-{((i + j) % 28) + 1:02d}",
                            "verified_purchase": j % 3 == 0,
                        }
                        for j in range(1, 8 + (i % 5))  # Variable reviews per product
                    ],
                    "tags": [f"tag_{(i + k) % 20}" for k in range(3)],
                }
                for i in range(1, 100)  # More products
            },
            "analytics": {
                "sales_by_month": {
                    f"2024-{month:02d}": {
                        "revenue": month * 12000 + (month % 3) * 5000,
                        "orders": month * 120 + (month % 4) * 30,
                        "customers": month * 25 + (month % 2) * 10,
                        "avg_order_value": (month * 12000 + (month % 3) * 5000)
                        / (month * 120 + (month % 4) * 30),
                        "top_category": ["Electronics", "Clothing", "Books", "Home"][
                            month % 4
                        ],
                    }
                    for month in range(1, 13)
                },
                "sales_by_category": {
                    category: {
                        "total_revenue": (i + 1) * 50000,
                        "total_orders": (i + 1) * 500,
                        "avg_rating": 3.5 + (i * 0.3),
                        "return_rate": 0.05 + (i * 0.02),
                        "top_products": [f"product_{j + i * 10}" for j in range(1, 6)],
                    }
                    for i, category in enumerate(
                        ["Electronics", "Clothing", "Books", "Home"]
                    )
                },
                "customer_segments": {
                    "high_value": {
                        "count": 25,
                        "avg_order": 350,
                        "total_lifetime_value": 25 * 350 * 4.2,
                        "avg_orders_per_customer": 4.2,
                        "preferred_categories": ["Electronics", "Home"],
                    },
                    "regular": {
                        "count": 50,
                        "avg_order": 150,
                        "total_lifetime_value": 50 * 150 * 2.8,
                        "avg_orders_per_customer": 2.8,
                        "preferred_categories": ["Clothing", "Books"],
                    },
                    "occasional": {
                        "count": 75,
                        "avg_order": 75,
                        "total_lifetime_value": 75 * 75 * 1.5,
                        "avg_orders_per_customer": 1.5,
                        "preferred_categories": ["Books", "Clothing"],
                    },
                },
                "geographic_data": {
                    f"region_{i}": {
                        "customers": 20 + (i * 5),
                        "revenue": (20 + (i * 5)) * 200,
                        "avg_shipping_cost": 15 + (i * 2),
                        "popular_categories": [
                            ["Electronics", "Home"],
                            ["Clothing", "Books"],
                            ["Books", "Electronics"],
                        ][i % 3],
                    }
                    for i in range(1, 8)
                },
            },
        }
    }


def create_complex_test_data():
    """Generate complex multi-domain test data"""
    return {
        "financial_data": {
            "companies": [
                {
                    "id": f"COMP{i:03d}",
                    "name": f"Company {i}",
                    "sector": [
                        "Technology",
                        "Healthcare",
                        "Finance",
                        "Energy",
                        "Retail",
                    ][i % 5],
                    "market_cap": (i * 1000000) + (i % 10) * 500000,
                    "employees": 100 + (i * 50) + (i % 20) * 25,
                    "quarterly_results": [
                        {
                            "quarter": f"Q{q}",
                            "year": 2024,
                            "revenue": (i * 10000) + (q * 5000) + ((i + q) % 1000),
                            "expenses": ((i * 10000) + (q * 5000) + ((i + q) % 1000))
                            * 0.75,
                            "profit_margin": 0.15 + ((i + q) % 10) * 0.02,
                            "employee_growth": ((i + q) % 15) - 7,  # -7 to +7 range
                        }
                        for q in range(1, 5)
                    ],
                    "stock_data": {
                        "current_price": 50 + (i % 200),
                        "52_week_high": 50 + (i % 200) + (i % 50),
                        "52_week_low": 50 + (i % 200) - (i % 30),
                        "volatility": 0.1 + ((i % 20) * 0.01),
                        "daily_prices": [
                            {
                                "date": f"2024-12-{day:02d}",
                                "open": 50 + (i % 200) + ((day + i) % 10),
                                "close": 50 + (i % 200) + ((day + i + 1) % 10),
                                "volume": 10000 + (day * 100) + (i % 1000),
                            }
                            for day in range(1, 29)
                        ],
                    },
                    "news_sentiment": [
                        {
                            "date": f"2024-{month:02d}-15",
                            "sentiment_score": -1
                            + ((i + month) % 20) * 0.1,  # -1 to 1 range
                            "news_count": 5 + ((i + month) % 15),
                            "topics": [
                                f"topic_{(i + month + j) % 25}" for j in range(3)
                            ],
                        }
                        for month in range(1, 13)
                    ],
                }
                for i in range(1, 250)  # 250 companies
            ],
            "market_indices": {
                "SP500": {
                    "current_value": 4500,
                    "daily_values": [
                        {
                            "date": f"2024-{month:02d}-{day:02d}",
                            "value": 4500 + (month * 50) + ((day + month) % 100) - 50,
                            "volume": 1000000 + (day * 10000),
                        }
                        for month in range(1, 13)
                        for day in range(1, min(29, 32))
                        if not (month == 2 and day > 28)
                    ],
                },
                "NASDAQ": {
                    "current_value": 14000,
                    "daily_values": [
                        {
                            "date": f"2024-{month:02d}-{day:02d}",
                            "value": 14000
                            + (month * 150)
                            + ((day + month) % 200)
                            - 100,
                            "volume": 2000000 + (day * 15000),
                        }
                        for month in range(1, 13)
                        for day in range(1, min(29, 32))
                        if not (month == 2 and day > 28)
                    ],
                },
            },
            "economic_indicators": {
                "interest_rates": [
                    {
                        "date": f"2024-{month:02d}-01",
                        "federal_rate": 5.0 + (month * 0.1) + ((month % 3) * 0.05),
                        "10_year_treasury": 4.5 + (month * 0.08) + ((month % 4) * 0.03),
                        "mortgage_30yr": 7.0 + (month * 0.12) + ((month % 2) * 0.1),
                    }
                    for month in range(1, 13)
                ],
                "inflation_data": [
                    {
                        "month": f"2024-{month:02d}",
                        "cpi": 100 + (month * 0.3) + ((month % 5) * 0.2),
                        "core_cpi": 100 + (month * 0.25) + ((month % 4) * 0.15),
                        "ppi": 100 + (month * 0.4) + ((month % 6) * 0.25),
                        "categories": {
                            cat: 100
                            + (month * ((i + 1) * 0.1))
                            + ((month % (i + 2)) * 0.1)
                            for i, cat in enumerate(
                                [
                                    "Housing",
                                    "Transportation",
                                    "Food",
                                    "Energy",
                                    "Medical",
                                ]
                            )
                        },
                    }
                    for month in range(1, 13)
                ],
            },
        },
        "social_media_data": {
            "platforms": {
                platform: {
                    "users": (i + 1) * 500000000,
                    "daily_posts": (i + 1) * 10000000,
                    "engagement_metrics": {
                        "avg_likes_per_post": 50 + (i * 20),
                        "avg_shares_per_post": 10 + (i * 5),
                        "avg_comments_per_post": 15 + (i * 8),
                    },
                    "trending_topics": [
                        {
                            "topic": f"#{platform}_trend_{j}",
                            "mentions": 100000 + (j * 10000) + ((i + j) % 50000),
                            "sentiment": -1 + ((i + j) % 20) * 0.1,
                            "geographic_spread": [
                                f"country_{(i + j + k) % 50}" for k in range(5)
                            ],
                        }
                        for j in range(1, 21)  # 20 trending topics per platform
                    ],
                }
                for i, platform in enumerate(
                    ["Twitter", "Facebook", "Instagram", "TikTok", "LinkedIn"]
                )
            },
            "influencers": [
                {
                    "id": f"influencer_{i}",
                    "username": f"@user{i}",
                    "followers": 10000 + (i * 5000) + ((i % 100) * 1000),
                    "platform": [
                        "Twitter",
                        "Facebook",
                        "Instagram",
                        "TikTok",
                        "LinkedIn",
                    ][i % 5],
                    "category": [
                        "Technology",
                        "Lifestyle",
                        "Business",
                        "Entertainment",
                        "Education",
                    ][i % 5],
                    "posts": [
                        {
                            "post_id": f"post_{i}_{j}",
                            "date": f"2024-{((j-1) // 30) + 1:02d}-{((j-1) % 30) + 1:02d}",
                            "content_type": ["text", "image", "video", "story"][j % 4],
                            "engagement": {
                                "likes": 100 + (i * 10) + (j * 5),
                                "shares": 10 + (i * 2) + (j * 1),
                                "comments": 20 + (i * 3) + (j * 2),
                            },
                            "reach": (100 + (i * 10) + (j * 5)) * 3,
                            "hashtags": [f"#tag{(i + j + k) % 100}" for k in range(3)],
                        }
                        for j in range(1, 366)  # Daily posts for a year
                    ],
                }
                for i in range(1, 500)  # 500 influencers
            ],
        },
    }


def parse_goals(goals_str: str) -> List[str]:
    """Parse comma-separated goals string into list of goals"""
    goal_str = str(goals_str)
    if not goal_str or len(goal_str.strip()) == 0:
        return []

    # Split by comma and clean up each goal
    goals = [goal.strip() for goal in goal_str.split(",")]
    # Remove empty goals
    goals = [goal for goal in goals if goal]

    return goals

@app.command(
    name="test",
    help="Run tests of the AI object navigation system with different datasets and analysis goals.",
)
def main(
    test_type: str | None = Option(
        None,
        "--type",
        "-t",
        help="Type of test to run: 'single' or 'multiple'. If not provided, will prompt interactively.",
    ),
    prompt: str | None = Option(
        None, "--prompt", "-p", help="Single analysis prompt/goal for single test mode."
    ),
    goals: str | None = Option(
        None,
        "--goals",
        "-g",
        help="Comma-separated list of analysis goals for multiple test mode. Example: 'Find top customers, Analyze sales trends, Identify growth opportunities'",
    ),
    json_input_path: str | None = Option(
        None,
        "--json-input",
        "-j",
        help="Path to a JSON file to use as input data instead of the default test data.",
    ),
    preview_depth: int = Option(
        5, "--preview-depth", "-d", help="Preview depth for object analysis."
    ),
    max_depth: int = Option(
        10, "--max-depth", "-m", help="Maximum depth for object analysis."
    ),
):
    """
    AI Object Navigation and Analysis Testing Tool.

    This command allows you to run tests of the AI object navigation system with
    different datasets and analysis goals.

    Examples:

    # Single analysis with custom data
    uv run scripts/test:main --json-input data.json --prompt "Find anomalies"

    # Multiple analyses with custom goals
    uv run scripts/test:main --type multiple --goals "Find top performers, Analyze trends, Identify opportunities"

    # Interactive mode (recommended for first time users)
    uv run scripts/test:main
    """

    # Handle conflicting options
    if prompt and goals:
        print(
            "⚠️  Warning: Both --prompt and --goals provided. Using --goals for multiple test mode."
        )
        test_type = "multiple"

    # If test_type not provided, ask interactively
    if test_type is None:
        test_options = ["single", "multiple"]
        print("🚀 AI Object Navigation Testing Tool")
        print("Select test type:")
        for i, option in enumerate(test_options, 1):
            description = (
                "Run one analysis with one goal"
                if option == "single"
                else "Run multiple analyses with multiple goals"
            )
            print(f"  {i}. {option.title()} Test - {description}")

        while True:
            try:
                choice = int(input("\nEnter your choice (1-2): "))
                if 1 <= choice <= 2:
                    test_type = test_options[choice - 1]
                    break
                else:
                    print("Invalid choice. Please enter 1 or 2.")
            except ValueError:
                print("Please enter a number.")

    # Handle prompts/goals based on test type
    custom_goals = []

    if test_type == "single":
        # Single test mode
        if prompt is None:
            use_custom = (
                input("\nDo you want to use a custom analysis prompt? (y/n): ")
                .lower()
                .startswith("y")
            )
            if use_custom:
                prompt = input("\nEnter your analysis prompt/goal: ")
        custom_goals = [prompt] if prompt else []

    else:  # multiple test mode
        if goals:
            # Goals provided via CLI
            print(f"\n📝 Using {goals} custom goals from command line")
            custom_goals = parse_goals(goals)
            print(f"\n📝 Using {len(custom_goals)} custom goals from command line")
        else:
            # Interactive goal input
            use_custom = (
                input("\nDo you want to use custom analysis goals? (y/n): ")
                .lower()
                .startswith("y")
            )
            if use_custom:
                print("\nEnter your analysis goals (comma-separated):")
                print(
                    "Example: Find top customers, Analyze sales trends, Identify growth opportunities"
                )
                goals_input = input("\nGoals: ")
                custom_goals = parse_goals(goals_input)

                if custom_goals:
                    print(f"\n📝 Parsed {len(custom_goals)} goals:")
                    for i, goal in enumerate(custom_goals, 1):
                        print(f"  {i}. {goal}")
                else:
                    print("No valid goals provided, will use defaults.")

    # Handle JSON input
    custom_data = None
    if json_input_path is None:
        use_json = (
            input(
                "\nDo you want to provide a JSON file as input instead of using test data? (y/n): "
            )
            .lower()
            .startswith("y")
        )
        if use_json:
            json_path = input("\nEnter the path to your JSON file: ")
            try:
                with open(json_path, "r") as f:
                    custom_data = json.load(f)
                print(
                    f"✅ Successfully loaded JSON data ({len(str(custom_data)):,} characters)"
                )
            except Exception as e:
                print(f"❌ Error loading JSON file: {e}")
                return
    else:
        try:
            with open(json_input_path, "r") as f:
                custom_data = json.load(f)
            print(
                f"✅ Successfully loaded JSON data from {json_input_path} ({len(str(custom_data)):,} characters)"
            )
        except Exception as e:
            print(f"❌ Error loading JSON file: {e}")
            return

    # Run appropriate test with asyncio.run
    if test_type == "single":
        asyncio.run(
            _run_single_test(
                custom_goals[0] if custom_goals else None,
                custom_data,
                preview_depth,
                max_depth,
            )
        )
    else:  # multiple
        asyncio.run(
            _run_multiple_test(custom_goals, custom_data, preview_depth, max_depth)
        )


async def _run_single_test(
    custom_prompt: Optional[str],
    custom_data: Optional[Dict[str, Any]],
    preview_depth: int,
    max_depth: int,
):
    """Run a single test with provided or default parameters"""
    # Auto-configure OpenAI
    get_settings()

    # Generate large test dataset if not provided
    if custom_data is None:
        large_data = create_large_ecommerce_data()
        print("📦 Using default e-commerce test dataset")
    else:
        large_data = custom_data
        print("📦 Using custom JSON dataset")

    print("🤖 Starting AI analysis of dataset...")
    print(f"📊 Dataset size: {len(str(large_data)):,} characters")

    if "e_commerce_data" in large_data:
        if "customers" in large_data["e_commerce_data"]:
            print(f"👥 Customers: {len(large_data['e_commerce_data']['customers'])}")
        if "products" in large_data["e_commerce_data"]:
            print(f"📦 Products: {len(large_data['e_commerce_data']['products'])}")

    # Create the workflow
    agent = ObjectAgent(timeout=300)

    # Use custom prompt or default
    goal = (
        custom_prompt
        or "Find the top 3 customers by total order value and analyze their purchasing patterns"
    )
    print(f"🎯 Analysis Goal: {goal}")

    # Create parameters
    params = ObjectAnalysisParams(
        data=large_data,
        goal=goal,
        max_depth=max_depth,
        preview_depth=preview_depth,
    )

    # Run the analysis
    result = await agent.run(start_event=StartEvent(params=params))

    # Display results
    if result.get("success"):
        print("\n✅ Analysis Complete!")
        print(f"🎯 Goal: {result['goal']}")
        if result.get("data_size_info"):
            print(f"📈 Data Overview: {result['data_size_info']}")
        print(f"\n🔍 AI Analysis:\n{'-' * 50}")
        print(result["analysis"])
        print(f"{'-' * 50}")
    else:
        print(f"❌ Analysis failed: {result.get('error')}")


async def _run_multiple_test(
    custom_goals: List[str],
    custom_data: Optional[Dict[str, Any]],
    preview_depth: int,
    max_depth: int,
):
    """Run multiple tests with provided or default parameters"""
    get_settings()

    # Create complex multi-domain test data if not provided
    if custom_data is None:
        test_data = create_complex_test_data()
        print("📦 Using default complex multi-domain test dataset")
    else:
        test_data = custom_data
        print("📦 Using custom JSON dataset")

    print(f"🧪 Complex Test Data:")
    print(f"📊 Dataset size: {len(str(test_data)):,} characters")

    # Print additional info if available in the standard format
    if "financial_data" in test_data and "companies" in test_data["financial_data"]:
        print(f"🏢 Companies: {len(test_data['financial_data']['companies'])}")
    if (
        "social_media_data" in test_data
        and "influencers" in test_data["social_media_data"]
    ):
        print(f"📱 Influencers: {len(test_data['social_media_data']['influencers'])}")

    # Use custom goals or defaults
    if custom_goals:
        goals = custom_goals
        print(f"\n📝 Running {len(goals)} custom analyses")
    else:
        goals = [
            "Identify the top 5 technology companies by profit margin growth and analyze their stock volatility patterns",
            "Find correlations between social media sentiment trends and stock price movements for companies in the dataset",
            "Analyze which economic indicators (interest rates, inflation) have the strongest correlation with market performance",
            "Identify the most successful social media influencers by engagement rate and determine what content types perform best",
            "Compare quarterly revenue growth patterns between different business sectors and predict which sectors show the most promise",
            "Analyze the relationship between company employee growth and stock performance across different market cap ranges",
        ]
        print(f"\n📝 Running {len(goals)} default analyses")

    agent = ObjectAgent(timeout=300)

    for i, goal in enumerate(goals, 1):
        print(f"\n🎯 Analysis {i}/{len(goals)}: {goal}")
        print("=" * 80)

        params = ObjectAnalysisParams(
            data=test_data, goal=goal, max_depth=max_depth, preview_depth=preview_depth
        )

        result = await agent.run(start_event=StartEvent(params=params))

        if result.get("success"):
            print(f"✅ Success!")
            # Show a preview of the analysis
            analysis_preview = (
                result["analysis"][:400] + "..."
                if len(result["analysis"]) > 400
                else result["analysis"]
            )
            print(f"Analysis Preview:\n{analysis_preview}")
            print(f"\n📊 Full analysis: {len(result['analysis'])} characters")
        else:
            print(f"❌ Failed: {result.get('error')}")

        print("\n" + "=" * 80)


async def test():
    """Main test function - entry point for UV"""
    await _run_single_test(None, None, 5, 10)


async def test_multiple_goals():
    """Test with different analysis goals on complex datasets"""
    await _run_multiple_test([], None, 4, 8)