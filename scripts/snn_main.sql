WITH matches AS (
    SELECT DISTINCT
        account_id,
        windfall_id ,
        candidate_id,
        confidence
    FROM `portal.match`),
    
# TODO: add employment features


accounts_opps as (
            select distinct
                a.Id as account_id,
                a.Member_Number__c as member_id,
                cast(PARSE_DATETIME("%Y-%m-%d", CloseDate) as date) as close_date,
                cast(Total_Purchase_Price_Fund_Program__c as float64) as fund_amount,
                case
                    when a.Member_Account_Type__c = 'Connect' then 'Connect'
                    when a.Member_Account_Type__c = 'Individual' then 'Core'
                    else 'Others'
                end as membership_type,
                cast(PARSE_DATETIME("%Y-%m-%d", Member_Start_Date__c) as date) as member_start_date
            from `wheelsup.sfdc_account` a
            left join `wheelsup.sfdc_opportunity` o on o.AccountId = a.Id
            where True
                and o.StageName = 'Closed Won'
                and a.Member_Account_Type__c in ('Individual', 'Connect')
                and a.IsDeleted = 'false'
                and o.IsDeleted = 'false'
                and a.Member_Number__c is not null
        )
        ,
        agg_funds as (
            select
                member_id,
                account_id,
                membership_type,
                sum(fund_amount) as total_funds,
                case
                    when sum(fund_amount) > 0 then True
                    else False
                end as is_funded
            from accounts_opps
            group by 1,2,3
        )
        ,
        wf_match as (
            select distinct
                a.*,
                c.How_Many_Private_Flights_Per_Year__c as private_flights, 
                c.id as contact_id,
                w.id as windfall_id,
                m.candidate_id as candidate_id,
                w.*,
                case
                    when w.id is not null then True
                    else False
                end as is_matched
            from agg_funds a
                left join  `wheelsup.sfdc_contact` c on  c.AccountId = a.account_id
                left join `portal.match` m on m.account_id = 385 and c.id = m.candidate_id and confidence >= 0.6
                left join `people.audience_latest` w on m.windfall_id = w.id
            where Primary_Contact__c = 'true'
        )
        ,
        dedupe_contacts as (
            with for_dedupe as (
                select *, row_number() over (partition by account_id) rn
                from wf_match
            )
            select * except (rn)
            from for_dedupe
            where rn=1
        ),

raw_data AS (
  SELECT DISTINCT
      CASE WHEN dedupe_contacts.membership_type IS NULL THEN "None" ELSE dedupe_contacts.membership_type END AS label,
      audience.* EXCEPT(recentDeathDate, recentFoundationAssociationDate, isFoundationOfficer, recentFoundationTrusteeDate, hasFoundationAssociation, hasCharityBoardMember, hasCharityOfficer,
                      isArtsCause,	isEducationCause,	isEnvironmentalCause,	isAnimalCause,	isHealthCause,	isHumanServicesCause,	isInternationalCause,	isSocialBenefitCause,	isReligiousCause,
                      isHouseholdDebt, isCharityBoardMember, isCharityOfficer, is990Donation,	isCoopDonation,	isFECContribution,	isStateContribution,
                      logMaxDonationAmount_1year,	logSumDonationAmount_1year,	logsumCOOPDonation_1year,	logsumFECDonation_1year,	logsumStateContribution_1year,	countNumCharities_1year, 
                      logMaxDonationAmount_3year,	logSumDonationAmount_3year,	logsum990Donation_3year,	logsumCOOPDonation_3year,	logsumFECDonation_3year,	logsumStateContribution_3year,	countNumCharities_3year,	
                      logMaxDonationAmount_5year,	logSumDonationAmount_5year,	logsum990Donation_5year,	logsumCOOPDonation_5year,	logsumFECDonation_5year,	logsumStateContribution_5year,	countNumCharities_5year,
                      lux_athletic, lux_flight, lux_goods, lux_travel, num_vehicles, num_luxury_vehicles, num_ultra_luxury_vehicles, num_cars, num_trucks, num_suvs, num_vans, metroRank),
      dbusa.realEstateInvestor,
      dbusa.personalInvestor,
      latest.city,
      latest.state,
      latest.zipcode,
      latest.county,
      latest.metroName,
      latest.censusPlaceFIPS,
      latest.primaryCarMake,
      latest.primaryCarModel,
      latest.isImportedCarOwner,
      latest.numberOfVehicles,
      FROM
      `tranquil-garage-139216.people.audience_latest` audience
      LEFT JOIN `tranquil-garage-139216.people.audience_dbusa_features` dbusa using(id)
      LEFT JOIN `tranquil-garage-139216.people.latest` latest ON latest.id = audience.id
      LEFT JOIN matches m ON audience.id = m.windfall_id
      LEFT JOIN dedupe_contacts ON dedupe_contacts.windfall_id = audience.id
      AND m.confidence > 0.90
      )
      
SELECT * EXCEPT(rn)
FROM (
  SELECT *,
  ROW_NUMBER() OVER(PARTITION BY label ORDER BY RAND()) AS rn
  FROM raw_data)
WHERE rn <= 7000
ORDER BY id, label