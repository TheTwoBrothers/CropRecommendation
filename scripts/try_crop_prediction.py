from app.services.crop_recommendation_service import (
    CropRecommendationService
)


def main():

    service = CropRecommendationService()

    input_features = {
        "SOIL": "sandy loamy soil",
        "SEASON": "zaid",
        "WATER_SOURCE": "irrigated",
        "TEMP": 28.5,
        "MAX_TEMP": 35,
        "RELATIVE_HUMIDITY": 78.3,
        "RELATIVE_HUMIDITY_MAX": 85
    }

    result = service.recommend(input_features)

    print("\n========== CROP RECOMMENDATIONS ==========")

    for item in result["recommendations"]:

        print(
            f"Rank {item['rank']}: "
            f"{item['crop']} | "
            f"Model Score: {item['model_score']}"
        )

    if result["warnings"]:
        print("\nWarnings:")

        for warning in result["warnings"]:
            print("-", warning)


if __name__ == "__main__":
    main()