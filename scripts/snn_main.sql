WITH matches AS (
    SELECT DISTINCT
        account_id,
        windfall_id ,
        candidate_id,
        confidence,
        CASE 
         -- luxury
         WHEN account_id = 81 THEN "1stDibs"
         WHEN account_id = 614 THEN "TamaraMellon"
         WHEN account_id = 585 THEN "Tonal"
         -- WHEN account_id = 385 THEN "WheelsUp"
         -- WHEN account_id = 208 THEN "Inspirato"
         -- WHEN account_id = 1577 THEN "OneFlight"
         -- alternative investment
         -- WHEN account_id = 501 THEN "Cadre"
         -- WHEN account_id = 679 THEN "Crowdstreet"
         -- WHEN account_id = 1047 THEN "Equaim"
         -- WHEN account_id = 1218 THEN "EquityEstates"
         -- WHEN account_id = 1246 THEN "EquityMultiple"
         WHEN account_id = 1050 THEN "MasterWorks"
         WHEN account_id = 753 THEN "Microventures"
         -- WHEN account_id = 1473 THEN "Portfolia"        
         -- insurance
         -- WHEN account_id = 514 THEN "HealthIQ"
         -- WHEN account_id = 1344 THEN "PureInsurance"
         -- finance
         -- WHEN account_id = 1219 THEN "SmartBiz"
         -- health
         -- WHEN account_id = 220 THEN "GrandViewHealth"
         -- WHEN account_id = 352 THEN "NewEnglandBaptistHospital"
         -- WHEN account_id = 1216 THEN "NuvanceHealth"
         -- WHEN account_id = 654 THEN "ProvidenceHealth"
         -- WHEN account_id = 1197 THEN "StCharles"
         END AS account_name,
         CASE 
         -- luxury
         WHEN account_id = 81 THEN "lux"
         WHEN account_id = 614 THEN "lux"
         WHEN account_id = 585 THEN "lux"
         -- WHEN account_id = 385 THEN "lux"
         -- WHEN account_id = 208 THEN "lux"
         -- WHEN account_id = 1577 THEN "lux"
         -- alternative investment
         -- WHEN account_id = 501 THEN "alt"
         -- WHEN account_id = 679 THEN "alt"
         -- WHEN account_id = 1047 THEN "alt"
         -- WHEN account_id = 1218 THEN "alt"
         -- WHEN account_id = 1246 THEN "alt"
         WHEN account_id = 1050 THEN "alt"
         WHEN account_id = 753 THEN "alt"
         -- WHEN account_id = 1473 THEN "alt"
         -- insurance
         -- WHEN account_id = 514 THEN "insurance"
         -- WHEN account_id = 1344 THEN "insurance"
         -- finance
         -- WHEN account_id = 1219 THEN "finance"
         -- health
         -- WHEN account_id = 220 THEN "health-donor"
         -- WHEN account_id = 352 THEN "health-donor"
         -- WHEN account_id = 1216 THEN "health-donor"
         -- WHEN account_id = 654 THEN "health-donor"
         -- WHEN account_id = 1197 THEN "health-donor"
         END AS label,
    FROM `portal.match`
    )

SELECT
    m.label,
    audience.*,
    latest.city,
    latest.state,
    latest.zipcode,
    latest.county,
    latest.metroName,
    realEstateInvestor,
    personalInvestor,
    FROM
    `tranquil-garage-139216.people.audience_latest` audience
    LEFT JOIN `tranquil-garage-139216.people.audience_dbusa_features` dbusa using(id)
    LEFT JOIN `tranquil-garage-139216.people.latest` latest ON latest.id = audience.id
    LEFT JOIN matches m ON audience.id = m.windfall_id
    WHERE m.label IS NOT NULL
    AND m.confidence > 0.90