# BSD-Analytics Data Quality Audit Report

**Status:** PASSED
**Total Records Analyzed:** 17,379

## 1. Schema Validation
- Missing Required Columns: 0 ([])
- Unexpected Extra Columns: 0 ([])

## 2. Completeness & Integrity
- Missing Values (Nulls): 0
- Duplicate Timestamps: 0
- Target (count) Negative Records: 0
- Target Identity Mismatches (casual + registered != count): 0

## 3. Physical Value Sanity Checks
- Out-of-bounds Temperatures ([-20C, 55C]): 0
- Out-of-bounds Humidity ([0%, 100%]): 0
- Out-of-bounds Windspeed ([0, 120 km/h]): 0

## 4. Anomaly Summary
- No critical anomalies detected. Dataset meets production quality standards.
