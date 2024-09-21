| FOCUS Dimension            |   Transform Step | Source Column               | Source Column Type   | Transform Type      | Filters/Process/Etc.   |
|:---------------------------|-----------------:|:----------------------------|:---------------------|:--------------------|:-----------------------|
| BillingAccountName         |                0 | Not Defined                 | Not Defined          | Not Defined         | Not Defined            |
| ChargeCategory             |                0 | Not Defined                 | Not Defined          | Not Defined         | Not Defined            |
| ChargeFrequency            |                0 | Not Defined                 | Not Defined          | Not Defined         | Not Defined            |
| CommitmentDiscountCategory |                0 | Not Defined                 | Not Defined          | Not Defined         | Not Defined            |
| CommitmentDiscountId       |                0 | Not Defined                 | Not Defined          | Not Defined         | Not Defined            |
| CommitmentDiscountName     |                0 | Not Defined                 | Not Defined          | Not Defined         | Not Defined            |
| CommitmentDiscountType     |                0 | Not Defined                 | Not Defined          | Not Defined         | Not Defined            |
| InvoiceIssuer              |                0 | Not Defined                 | Not Defined          | Not Defined         | Not Defined            |
| ListCost                   |                0 | Not Defined                 | Not Defined          | Not Defined         | Not Defined            |
| ListUnitPrice              |                0 | Not Defined                 | Not Defined          | Not Defined         | Not Defined            |
| PricingCategory            |                0 | Not Defined                 | Not Defined          | Not Defined         | Not Defined            |
| PricingQuantity            |                0 | Not Defined                 | Not Defined          | Not Defined         | Not Defined            |
| Publisher                  |                0 | Not Defined                 | Not Defined          | Not Defined         | Not Defined            |
| ResourceName               |                0 | Not Defined                 | Not Defined          | Not Defined         | Not Defined            |
| ResourceType               |                0 | Not Defined                 | Not Defined          | Not Defined         | Not Defined            |
| ServiceCategory            |                0 | Not Defined                 | Not Defined          | Not Defined         | Not Defined            |
| SkuPriceId                 |                0 | Not Defined                 | Not Defined          | Not Defined         | Not Defined            |
| SubAccountName             |                0 | Not Defined                 | Not Defined          | Not Defined         | Not Defined            |
| AvailabilityZone           |                1 | product/availabilityDomain  | Not Defined          | RENAME_COLUMN       |                        |
| BilledCost                 |                1 | cost/myCostOverage          | Not Defined          | RENAME_COLUMN       |                        |
| BillingAccountId           |                1 | cost/subscriptionId         | Not Defined          | RENAME_COLUMN       |                        |
| BillingCurrency            |                1 | cost/currencyCode           | Not Defined          | RENAME_COLUMN       |                        |
| BillingPeriodEnd           |                1 | lineItem/intervalUsageEnd   | datetime             | MONTH_END           |                        |
| BillingPeriodStart         |                1 | lineItem/intervalUsageStart | datetime             | MONTH_START         |                        |
| ChargeDescription          |                1 | product/Description         | Not Defined          | RENAME_COLUMN       |                        |
| ChargePeriodEnd            |                1 | lineItem/intervalUsageEnd   | datetime             | RENAME_COLUMN       |                        |
| ChargePeriodStart          |                1 | lineItem/intervalUsageStart | datetime             | RENAME_COLUMN       |                        |
| ConsumedQuantity           |                1 | usage/billedQuantity        | Not Defined          | RENAME_COLUMN       |                        |
| ConsumedUnit               |                1 | cost/skuUnitDescription     | Not Defined          | RENAME_COLUMN       |                        |
| EffectiveCost              |                1 | cost/myCost                 | Not Defined          | RENAME_COLUMN       |                        |
| PricingUnit                |                1 | cost/skuUnitDescription     | Not Defined          | RENAME_COLUMN       |                        |
| Provider                   |                1 | NA                          | Not Defined          | ASSIGN_STATIC_VALUE | static_value: Oracle   |
| RegionId                   |                1 | product/region              | Not Defined          | RENAME_COLUMN       |                        |
| ResourceId                 |                1 | product/resourceId          | Not Defined          | RENAME_COLUMN       |                        |
| ServiceName                |                1 | product/service             | Not Defined          | RENAME_COLUMN       |                        |
| SkuId                      |                1 | cost/productSku             | Not Defined          | RENAME_COLUMN       |                        |
| SubAccountId               |                1 | lineItem/tenantId           | Not Defined          | RENAME_COLUMN       |                        |