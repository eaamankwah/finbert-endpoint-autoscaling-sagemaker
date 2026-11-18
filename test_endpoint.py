import argparse
import json
import boto3


def test_endpoint(endpoint_name: str, region: str = None):
    # Automatically detects the region from your AWS environment if not provided
    session = boto3.Session(region_name=region)
    runtime = session.client("sagemaker-runtime")

    test_inputs = [
        # --- Strong Positive ---
        "The company reported record profits and strong revenue growth.",
        "Quarterly net income surged 35%, beating Wall Street consensus estimates.",
        "The board approved a 15% dividend increase and a $2 billion share buyback.",
        "Shares rallied after the firm raised its full-year operating margin guidance.",
        # --- Strong Negative ---
        "The stock plummeted after missing earnings expectations.",
        "Operating losses widened sharply due to supply chain disruptions and rising debt.",
        "Regulators launched an antitrust investigation, sending shares down 12%.",
        "The company announced layoffs of 10% of its workforce amid slowing demand.",
        # --- Neutral / Factual ---
        "Trading volume remained steady with no major changes.",
        "The annual general meeting will be held next Tuesday at the headquarters.",
        "The company appointed a new Chief Financial Officer effective next month.",
        "Management reaffirmed its previously announced third-quarter revenue forecast.",
        # --- Nuanced / Financial Jargon ---
        "Revenue grew modestly, though higher input costs pressured gross margins.",
        "The acquisition is expected to be accretive to earnings per share by next year.",
        "Central bank officials signaled interest rates will remain unchanged for now.",
    ]

    print(
        f"Sending {len(test_inputs)} test requests to endpoint: {endpoint_name}..."
    )

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps({"inputs": test_inputs}),
    )

    result = json.loads(response["Body"].read().decode("utf-8"))
    id2label = {0: "negative", 1: "neutral", 2: "positive"}

    print("\nPredictions:")
    for text, pred in zip(test_inputs, result):
        # Handles both numeric string labels (e.g., "2") and text labels (e.g., "positive")
        raw_label = str(pred["label"]).replace("LABEL_", "")
        label = (
            id2label.get(int(raw_label), raw_label)
            if raw_label.isdigit()
            else raw_label
        )
        print(
            f"  [Sentiment: {label:8} - Confidence: {pred['score']:.2f}]  {text[:65]}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test deployed FinBERT SageMaker endpoint"
    )
    parser.add_argument(
        "--endpoint-name",
        default="finbert-sentiment-endpoint",
        help="Name of the deployed SageMaker endpoint",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region (defaults to your current environment region)",
    )
    args = parser.parse_args()

    test_endpoint(args.endpoint_name, args.region)
