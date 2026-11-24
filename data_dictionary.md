# Data Dictionary

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| date | datetime | Observation date | Sales DB |
| sales | float | Units sold (target) | Sales DB |
| promo_flag | int | 1 if promotion active | Promo calendar |
| holiday_flag | int | 1 if public holiday | Holiday API |
| sentiment_score | float | Social media sentiment (-1 to 1) | Twitter API |
| gdp_index | float | Economic indicator | World Bank |