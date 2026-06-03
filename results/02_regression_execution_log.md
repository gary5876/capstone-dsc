# 노트북 02 회귀 버전 실행 로그

- **실행 시각**: 2026-06-03 09:44:47
- **총 실험**: 81건 (베이스라인 포함)
- **소요 시간**: 151초
- **에러**: 0건

## 1. 오염 설정

- **데이터셋**: ['CaliforniaHousing', 'BikeSharing', 'WineQuality']
- **오염 강도**: [0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
- **Polluter**: Completeness, Uniqueness, FeatureAccuracy, ConsistentRepresentation(범주형 데이터셋만), TargetDistributionSkew

## 2. DSC 점수 결과 (회귀 cell)

| 데이터셋 | 오염 유형 | 강도 | DSC | 등급 | completeness | uniqueness | validity | consistency | outlier_ratio | target_distribution_quality | feature_correlation | target_smoothness | feature_informativeness_reg |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CaliforniaHousing | none | 0.0 | 88.73 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.9724 | 0.9122 | 1.0 | 0.6955 | 0.5836 |
| CaliforniaHousing | completeness | 0.1 | 83.83 | B | 0.9 | 1.0 | 1.0 | 1.0 | 0.9003 | 0.9122 | 1.0 | 0.6385 | 0.444 |
| CaliforniaHousing | uniqueness | 0.1 | 83.61 | B | 1.0 | 0.6406 | 1.0 | 1.0 | 0.9722 | 0.9177 | 1.0 | 0.7206 | 0.5548 |
| CaliforniaHousing | feature_accuracy | 0.1 | 84.02 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.8799 | 0.9122 | 1.0 | 0.6419 | 0.2656 |
| CaliforniaHousing | target_distribution_skew | 0.1 | 88.61 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.9726 | 0.9025 | 1.0 | 0.7136 | 0.5447 |
| CaliforniaHousing | completeness | 0.25 | 78.99 | B | 0.75 | 1.0 | 1.0 | 1.0 | 0.792 | 0.9122 | 1.0 | 0.6063 | 0.3782 |
| CaliforniaHousing | uniqueness | 0.25 | 81.6 | B | 1.0 | 0.5 | 1.0 | 1.0 | 0.9725 | 0.9122 | 1.0 | 0.7163 | 0.5788 |
| CaliforniaHousing | feature_accuracy | 0.25 | 81.7 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.7443 | 0.9122 | 1.0 | 0.6075 | 0.1706 |
| CaliforniaHousing | target_distribution_skew | 0.25 | 88.74 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.9727 | 0.8865 | 1.0 | 0.7186 | 0.5641 |
| CaliforniaHousing | completeness | 0.5 | 71.43 | C | 0.5 | 0.9999 | 1.0 | 1.0 | 0.6115 | 0.9122 | 1.0 | 0.582 | 0.2614 |
| CaliforniaHousing | uniqueness | 0.5 | 78.82 | B | 1.0 | 0.3333 | 1.0 | 1.0 | 0.9727 | 0.9122 | 1.0 | 0.6993 | 0.5848 |
| CaliforniaHousing | feature_accuracy | 0.5 | 79.39 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.6044 | 0.9122 | 1.0 | 0.5718 | 0.0811 |
| CaliforniaHousing | target_distribution_skew | 0.5 | 87.71 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.9732 | 0.8499 | 1.0 | 0.7025 | 0.5295 |
| CaliforniaHousing | completeness | 0.75 | 63.38 | C | 0.25 | 0.9706 | 1.0 | 1.0 | 0.4306 | 0.9122 | 1.0 | 0.5603 | 0.1339 |
| CaliforniaHousing | uniqueness | 0.75 | 77.85 | B | 1.0 | 0.25 | 1.0 | 1.0 | 0.9727 | 0.9122 | 1.0 | 0.7147 | 0.5819 |
| CaliforniaHousing | feature_accuracy | 0.75 | 78.42 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.5019 | 0.9122 | 1.0 | 0.557 | 0.0649 |
| CaliforniaHousing | target_distribution_skew | 0.75 | 87.14 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.9739 | 0.7952 | 1.0 | 0.6886 | 0.5546 |
| CaliforniaHousing | completeness | 0.9 | 55.53 | D | 0.1 | 0.7386 | 1.0 | 1.0 | 0.3221 | 0.9122 | 1.0 | 0.5535 | 0.0648 |
| CaliforniaHousing | uniqueness | 0.9 | 76.86 | B | 1.0 | 0.2 | 1.0 | 1.0 | 0.9724 | 0.9122 | 1.0 | 0.7114 | 0.5643 |
| CaliforniaHousing | feature_accuracy | 0.9 | 77.9 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.4531 | 0.9122 | 1.0 | 0.5515 | 0.0484 |
| CaliforniaHousing | target_distribution_skew | 0.9 | 86.44 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.9742 | 0.7449 | 1.0 | 0.6812 | 0.5499 |
| CaliforniaHousing | completeness | 0.95 | 51.14 | D | 0.05 | 0.5351 | 1.0 | 1.0 | 0.2862 | 0.9122 | 1.0 | 0.5576 | 0.0406 |
| CaliforniaHousing | uniqueness | 0.95 | 76.8 | B | 1.0 | 0.1667 | 1.0 | 1.0 | 0.9726 | 0.9122 | 1.0 | 0.7189 | 0.5939 |
| CaliforniaHousing | feature_accuracy | 0.95 | 77.69 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.4378 | 0.9122 | 1.0 | 0.5511 | 0.0352 |
| CaliforniaHousing | target_distribution_skew | 0.95 | 86.21 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.9744 | 0.7231 | 1.0 | 0.6828 | 0.5446 |
| BikeSharing | none | 0.0 | 87.31 | B | 1.0 | 0.9999 | 1.0 | 1.0 | 0.9971 | 0.7284 | 0.9524 | 0.6734 | 0.6809 |
| BikeSharing | completeness | 0.1 | 84.31 | B | 0.9 | 1.0 | 1.0 | 1.0 | 0.9403 | 0.7284 | 1.0 | 0.652 | 0.6286 |
| BikeSharing | uniqueness | 0.1 | 82.55 | B | 1.0 | 0.6592 | 1.0 | 1.0 | 0.9972 | 0.735 | 0.9524 | 0.6755 | 0.7054 |
| BikeSharing | feature_accuracy | 0.1 | 85.22 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.9969 | 0.7284 | 1.0 | 0.6574 | 0.4806 |
| BikeSharing | consistent_repr | 0.1 | 86.56 | B | 1.0 | 1.0 | 1.0 | 0.9 | 0.9971 | 0.7284 | 0.9524 | 0.6734 | 0.7059 |
| BikeSharing | target_distribution_skew | 0.1 | 87.34 | B | 1.0 | 0.9999 | 1.0 | 1.0 | 0.997 | 0.714 | 0.9524 | 0.6712 | 0.7031 |
| BikeSharing | completeness | 0.25 | 78.83 | B | 0.7501 | 1.0 | 1.0 | 1.0 | 0.855 | 0.7284 | 1.0 | 0.6015 | 0.5245 |
| BikeSharing | uniqueness | 0.25 | 80.35 | B | 1.0 | 0.5 | 1.0 | 1.0 | 0.997 | 0.7284 | 0.9524 | 0.6716 | 0.7387 |
| BikeSharing | feature_accuracy | 0.25 | 83.74 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.9941 | 0.7284 | 1.0 | 0.6284 | 0.3919 |
| BikeSharing | consistent_repr | 0.25 | 84.79 | B | 1.0 | 0.9999 | 1.0 | 0.7501 | 0.9971 | 0.7284 | 0.9524 | 0.6734 | 0.679 |
| BikeSharing | target_distribution_skew | 0.25 | 87.79 | B | 1.0 | 0.9999 | 1.0 | 1.0 | 0.997 | 0.6896 | 0.9524 | 0.6802 | 0.7548 |
| BikeSharing | completeness | 0.5 | 70.66 | C | 0.5 | 1.0 | 1.0 | 1.0 | 0.7128 | 0.7284 | 1.0 | 0.572 | 0.337 |
| BikeSharing | uniqueness | 0.5 | 76.69 | B | 1.0 | 0.3333 | 1.0 | 1.0 | 0.997 | 0.7284 | 0.9524 | 0.6664 | 0.6333 |
| BikeSharing | feature_accuracy | 0.5 | 80.91 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.9628 | 0.7284 | 1.0 | 0.5884 | 0.2049 |
| BikeSharing | consistent_repr | 0.5 | 82.46 | B | 1.0 | 1.0 | 1.0 | 0.5 | 0.9971 | 0.7284 | 0.9524 | 0.6734 | 0.6961 |
| BikeSharing | target_distribution_skew | 0.5 | 87.8 | B | 1.0 | 0.9999 | 1.0 | 1.0 | 0.9969 | 0.6384 | 0.9524 | 0.6847 | 0.7981 |
| BikeSharing | completeness | 0.75 | 63.25 | C | 0.25 | 0.98 | 1.0 | 1.0 | 0.5707 | 0.7284 | 1.0 | 0.5881 | 0.1654 |
| BikeSharing | uniqueness | 0.75 | 76.56 | B | 1.0 | 0.25 | 1.0 | 1.0 | 0.9969 | 0.7284 | 0.9524 | 0.6713 | 0.7358 |
| BikeSharing | feature_accuracy | 0.75 | 79.39 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.8982 | 0.7284 | 1.0 | 0.5693 | 0.1231 |
| BikeSharing | consistent_repr | 0.75 | 79.9 | B | 1.0 | 1.0 | 1.0 | 0.25 | 0.9971 | 0.7284 | 0.9524 | 0.6734 | 0.6906 |
| BikeSharing | target_distribution_skew | 0.75 | 86.01 | B | 1.0 | 0.9999 | 1.0 | 1.0 | 0.9969 | 0.5644 | 0.9524 | 0.6795 | 0.7026 |
| BikeSharing | completeness | 0.9 | 54.81 | D | 0.1001 | 0.7171 | 1.0 | 1.0 | 0.4855 | 0.7284 | 1.0 | 0.5909 | 0.0522 |
| BikeSharing | uniqueness | 0.9 | 75.12 | B | 1.0 | 0.2 | 1.0 | 1.0 | 0.997 | 0.7284 | 0.9524 | 0.6694 | 0.67 |
| BikeSharing | feature_accuracy | 0.9 | 78.71 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.8569 | 0.7284 | 1.0 | 0.5617 | 0.0909 |
| BikeSharing | consistent_repr | 0.9 | 78.23 | B | 1.0 | 0.9999 | 1.0 | 0.1001 | 0.9971 | 0.7284 | 0.9524 | 0.6734 | 0.6732 |
| BikeSharing | target_distribution_skew | 0.9 | 86.91 | B | 1.0 | 0.9999 | 1.0 | 1.0 | 0.9968 | 0.502 | 0.9524 | 0.686 | 0.8421 |
| BikeSharing | completeness | 0.95 | 50.1 | D | 0.0501 | 0.4709 | 1.0 | 1.0 | 0.457 | 0.7284 | 1.0 | 0.5912 | 0.0643 |
| BikeSharing | uniqueness | 0.95 | 75.92 | B | 1.0 | 0.1667 | 1.0 | 1.0 | 0.9972 | 0.7284 | 0.9524 | 0.6791 | 0.7802 |
| BikeSharing | feature_accuracy | 0.95 | 78.74 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.8398 | 0.7284 | 1.0 | 0.5634 | 0.099 |
| BikeSharing | consistent_repr | 0.95 | 77.79 | B | 1.0 | 0.9999 | 1.0 | 0.0501 | 0.9971 | 0.7284 | 0.9524 | 0.6734 | 0.679 |
| BikeSharing | target_distribution_skew | 0.95 | 84.99 | B | 1.0 | 0.9999 | 1.0 | 1.0 | 0.9968 | 0.473 | 0.9524 | 0.6692 | 0.713 |
| WineQuality | none | 0.0 | 79.08 | B | 1.0 | 0.843 | 1.0 | 1.0 | 0.973 | 0.5552 | 1.0 | 0.6008 | 0.4004 |
| WineQuality | completeness | 0.1 | 78.88 | B | 0.9001 | 0.9802 | 1.0 | 1.0 | 0.9027 | 0.5552 | 1.0 | 0.5616 | 0.4881 |
| WineQuality | uniqueness | 0.1 | 76.2 | B | 1.0 | 0.6666 | 1.0 | 1.0 | 0.972 | 0.5572 | 1.0 | 0.5974 | 0.382 |
| WineQuality | feature_accuracy | 0.1 | 78.52 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.8647 | 0.5552 | 1.0 | 0.5833 | 0.1974 |
| WineQuality | target_distribution_skew | 0.1 | 78.58 | B | 1.0 | 0.8496 | 1.0 | 1.0 | 0.9724 | 0.5571 | 1.0 | 0.603 | 0.3348 |
| WineQuality | completeness | 0.25 | 74.08 | C | 0.75 | 0.999 | 1.0 | 1.0 | 0.7977 | 0.5552 | 1.0 | 0.5492 | 0.357 |
| WineQuality | uniqueness | 0.25 | 73.86 | C | 1.0 | 0.5 | 1.0 | 1.0 | 0.9716 | 0.5572 | 1.0 | 0.6065 | 0.3805 |
| WineQuality | feature_accuracy | 0.25 | 76.33 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.7875 | 0.5552 | 1.0 | 0.5578 | 0.0687 |
| WineQuality | target_distribution_skew | 0.25 | 79.01 | B | 1.0 | 0.8632 | 1.0 | 1.0 | 0.9719 | 0.5575 | 1.0 | 0.6098 | 0.3427 |
| WineQuality | completeness | 0.5 | 66.85 | C | 0.5001 | 0.9998 | 1.0 | 1.0 | 0.6233 | 0.5552 | 1.0 | 0.5382 | 0.2417 |
| WineQuality | uniqueness | 0.5 | 71.09 | C | 1.0 | 0.3333 | 1.0 | 1.0 | 0.972 | 0.5572 | 1.0 | 0.6016 | 0.3625 |
| WineQuality | feature_accuracy | 0.5 | 75.93 | B | 1.0 | 1.0 | 1.0 | 1.0 | 0.6747 | 0.5552 | 1.0 | 0.5437 | 0.1131 |
| WineQuality | target_distribution_skew | 0.5 | 79.91 | B | 1.0 | 0.8822 | 1.0 | 1.0 | 0.9706 | 0.5449 | 1.0 | 0.6106 | 0.4167 |
| WineQuality | completeness | 0.75 | 59.05 | D | 0.2501 | 0.928 | 1.0 | 1.0 | 0.4476 | 0.5552 | 1.0 | 0.541 | 0.1521 |
| WineQuality | uniqueness | 0.75 | 70.07 | C | 1.0 | 0.25 | 1.0 | 1.0 | 0.9724 | 0.5572 | 1.0 | 0.6087 | 0.3717 |
| WineQuality | feature_accuracy | 0.75 | 74.98 | C | 1.0 | 1.0 | 1.0 | 1.0 | 0.5816 | 0.5552 | 1.0 | 0.5367 | 0.0782 |
| WineQuality | target_distribution_skew | 0.75 | 79.32 | B | 1.0 | 0.8861 | 1.0 | 1.0 | 0.9688 | 0.4803 | 1.0 | 0.6199 | 0.3988 |
| WineQuality | completeness | 0.9 | 49.75 | D | 0.1001 | 0.5619 | 1.0 | 1.0 | 0.3428 | 0.5552 | 1.0 | 0.5476 | 0.1103 |
| WineQuality | uniqueness | 0.9 | 69.24 | C | 1.0 | 0.2 | 1.0 | 1.0 | 0.9719 | 0.5572 | 1.0 | 0.604 | 0.3729 |
| WineQuality | feature_accuracy | 0.9 | 74.6 | C | 1.0 | 1.0 | 1.0 | 1.0 | 0.5338 | 0.5552 | 1.0 | 0.5369 | 0.0643 |
| WineQuality | target_distribution_skew | 0.9 | 77.44 | B | 1.0 | 0.8752 | 1.0 | 1.0 | 0.9673 | 0.3676 | 1.0 | 0.6364 | 0.3068 |
| WineQuality | completeness | 0.95 | 44.34 | D | 0.05 | 0.3288 | 1.0 | 1.0 | 0.3075 | 0.5552 | 1.0 | 0.5469 | 0.0383 |
| WineQuality | uniqueness | 0.95 | 68.63 | C | 1.0 | 0.1667 | 1.0 | 1.0 | 0.9723 | 0.5572 | 1.0 | 0.5971 | 0.3759 |
| WineQuality | feature_accuracy | 0.95 | 74.39 | C | 1.0 | 1.0 | 1.0 | 1.0 | 0.5183 | 0.5552 | 1.0 | 0.5354 | 0.0535 |
| WineQuality | target_distribution_skew | 0.95 | 78.42 | B | 1.0 | 0.87 | 1.0 | 1.0 | 0.9656 | 0.2958 | 1.0 | 0.6539 | 0.4504 |
