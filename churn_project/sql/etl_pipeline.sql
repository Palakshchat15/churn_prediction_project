-- =============================================================================
-- churn_project/sql/etl_pipeline.sql
-- 100% SQL-based Analytics Engineering for the Cell2Cell churn dataset.
-- =============================================================================

-- FINAL CLEANING QUERY
-- Note: The staging table is created dynamically by the Python runner.
SELECT 
    -- ID and Target
    CustomerID,
    CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END AS Churn,
    
    -- Numerical Casting
    TRY_CAST(MonthlyRevenue AS FLOAT) AS MonthlyRevenue,
    TRY_CAST(MonthlyMinutes AS FLOAT) AS MonthlyMinutes,
    TRY_CAST(TotalRecurringCharge AS FLOAT) AS TotalRecurringCharge,
    TRY_CAST(DirectorAssistedCalls AS FLOAT) AS DirectorAssistedCalls,
    TRY_CAST(OverageMinutes AS FLOAT) AS OverageMinutes,
    TRY_CAST(RoamingCalls AS FLOAT) AS RoamingCalls,
    TRY_CAST(PercChangeMinutes AS FLOAT) AS PercChangeMinutes,
    TRY_CAST(PercChangeRevenues AS FLOAT) AS PercChangeRevenues,
    TRY_CAST(DroppedCalls AS FLOAT) AS DroppedCalls,
    TRY_CAST(BlockedCalls AS FLOAT) AS BlockedCalls,
    TRY_CAST(UnansweredCalls AS FLOAT) AS UnansweredCalls,
    TRY_CAST(CustomerCareCalls AS FLOAT) AS CustomerCareCalls,
    TRY_CAST(ThreewayCalls AS FLOAT) AS ThreewayCalls,
    TRY_CAST(ReceivedCalls AS FLOAT) AS ReceivedCalls,
    TRY_CAST(OutboundCalls AS FLOAT) AS OutboundCalls,
    TRY_CAST(InboundCalls AS FLOAT) AS InboundCalls,
    TRY_CAST(PeakCallsInOut AS FLOAT) AS PeakCallsInOut,
    TRY_CAST(OffPeakCallsInOut AS FLOAT) AS OffPeakCallsInOut,
    TRY_CAST(DroppedBlockedCalls AS FLOAT) AS DroppedBlockedCalls,
    TRY_CAST(CallForwardingCalls AS FLOAT) AS CallForwardingCalls,
    TRY_CAST(CallWaitingCalls AS FLOAT) AS CallWaitingCalls,
    TRY_CAST(MonthsInService AS FLOAT) AS MonthsInService,
    TRY_CAST(UniqueSubs AS FLOAT) AS UniqueSubs,
    TRY_CAST(ActiveSubs AS FLOAT) AS ActiveSubs,
    TRY_CAST(Handsets AS FLOAT) AS Handsets,
    TRY_CAST(HandsetModels AS FLOAT) AS HandsetModels,
    TRY_CAST(CurrentEquipmentDays AS FLOAT) AS CurrentEquipmentDays,
    TRY_CAST(AgeHH1 AS FLOAT) AS AgeHH1,
    TRY_CAST(AgeHH2 AS FLOAT) AS AgeHH2,
    TRY_CAST(HandsetPrice AS FLOAT) AS HandsetPrice,
    TRY_CAST(RetentionCalls AS FLOAT) AS RetentionCalls,
    TRY_CAST(RetentionOffersAccepted AS FLOAT) AS RetentionOffersAccepted,
    TRY_CAST(ReferralsMadeBySubscriber AS FLOAT) AS ReferralsMadeBySubscriber,
    TRY_CAST(IncomeGroup AS FLOAT) AS IncomeGroup,
    TRY_CAST(AdjustmentsToCreditRating AS FLOAT) AS AdjustmentsToCreditRating,

    -- Binary Encodings
    CASE WHEN ChildrenInHH = 'Yes' THEN 1 ELSE 0 END AS ChildrenInHH,
    CASE WHEN HandsetRefurbished = 'Yes' THEN 1 ELSE 0 END AS HandsetRefurbished,
    CASE WHEN HandsetWebCapable = 'Yes' THEN 1 ELSE 0 END AS HandsetWebCapable,
    CASE WHEN TruckOwner = 'Yes' THEN 1 ELSE 0 END AS TruckOwner,
    CASE WHEN RVOwner = 'Yes' THEN 1 ELSE 0 END AS RVOwner,
    CASE WHEN Homeownership = 'Known' THEN 1 ELSE 0 END AS Homeownership,
    CASE WHEN BuysViaMailOrder = 'Yes' THEN 1 ELSE 0 END AS BuysViaMailOrder,
    CASE WHEN RespondsToMailOffers = 'Yes' THEN 1 ELSE 0 END AS RespondsToMailOffers,
    CASE WHEN OptOutMailings = 'Yes' THEN 1 ELSE 0 END AS OptOutMailings,
    CASE WHEN NonUSTravel = 'Yes' THEN 1 ELSE 0 END AS NonUSTravel,
    CASE WHEN OwnsComputer = 'Yes' THEN 1 ELSE 0 END AS OwnsComputer,
    CASE WHEN HasCreditCard = 'Yes' THEN 1 ELSE 0 END AS HasCreditCard,
    CASE WHEN NewCellphoneUser = 'Yes' THEN 1 ELSE 0 END AS NewCellphoneUser,
    CASE WHEN OwnsMotorcycle = 'Yes' THEN 1 ELSE 0 END AS OwnsMotorcycle,
    CASE WHEN MadeCallToRetentionTeam = 'Yes' THEN 1 ELSE 0 END AS MadeCallToRetentionTeam,

    -- Categorical / Ordinal
    CASE 
        WHEN CreditRating = '1-Highest' THEN 1
        WHEN CreditRating = '2-High' THEN 2
        WHEN CreditRating = '3-Good' THEN 3
        WHEN CreditRating = '4-Medium' THEN 4
        WHEN CreditRating = '5-Low' THEN 5
        WHEN CreditRating = '6-VeryLow' THEN 6
        WHEN CreditRating = '7-Lowest' THEN 7
        ELSE 4
    END AS CreditRating,
    
    CASE 
        WHEN MaritalStatus = 'Yes' THEN 1 
        WHEN MaritalStatus = 'No' THEN 0 
        ELSE -1 
    END AS MaritalStatus,

    -- ADVANCED ANALYTICS ENGINEERING
    (TRY_CAST(DroppedCalls AS FLOAT) + TRY_CAST(BlockedCalls AS FLOAT)) / NULLIF(TRY_CAST(MonthlyMinutes AS FLOAT), 0) AS DropRate,
    TRY_CAST(BlockedCalls AS FLOAT) / NULLIF(TRY_CAST(MonthlyMinutes AS FLOAT), 0) AS BlockRate,
    TRY_CAST(OverageMinutes AS FLOAT) / NULLIF(TRY_CAST(MonthlyMinutes AS FLOAT), 0) AS OverageRatio,
    TRY_CAST(MonthlyRevenue AS FLOAT) / NULLIF(TRY_CAST(MonthlyMinutes AS FLOAT), 0) AS RevenuePerMinute,
    TRY_CAST(CurrentEquipmentDays AS FLOAT) / 365.25 AS EquipmentAgeYears,
    (TRY_CAST(AgeHH1 AS FLOAT) + TRY_CAST(AgeHH2 AS FLOAT)) / 2 AS AvgHouseholdAge,
    
    CASE 
        WHEN TRY_CAST(MonthsInService AS FLOAT) <= 6 THEN 'New'
        WHEN TRY_CAST(MonthsInService AS FLOAT) <= 24 THEN 'Loyal'
        ELSE 'Veteran'
    END AS TenureBucket,

    -- One-Hot Encoding for PrizmCode
    CASE WHEN PrizmCode = 'Other' THEN 1 ELSE 0 END AS PrizmCode_Other,
    CASE WHEN PrizmCode = 'Rural' THEN 1 ELSE 0 END AS PrizmCode_Rural,
    CASE WHEN PrizmCode = 'Suburban' THEN 1 ELSE 0 END AS PrizmCode_Suburban,
    CASE WHEN PrizmCode = 'Town' THEN 1 ELSE 0 END AS PrizmCode_Town,

    -- One-Hot Encoding for Occupation
    CASE WHEN Occupation = 'Clerical' THEN 1 ELSE 0 END AS Occupation_Clerical,
    CASE WHEN Occupation = 'Crafts' THEN 1 ELSE 0 END AS Occupation_Crafts,
    CASE WHEN Occupation = 'Homemaker' THEN 1 ELSE 0 END AS Occupation_Homemaker,
    CASE WHEN Occupation = 'Other' THEN 1 ELSE 0 END AS Occupation_Other,
    CASE WHEN Occupation = 'Professional' THEN 1 ELSE 0 END AS Occupation_Professional,
    CASE WHEN Occupation = 'Retired' THEN 1 ELSE 0 END AS Occupation_Retired,
    CASE WHEN Occupation = 'Self' THEN 1 ELSE 0 END AS Occupation_Self,
    CASE WHEN Occupation = 'Student' THEN 1 ELSE 0 END AS Occupation_Student

FROM staging
WHERE Handsets IS NOT NULL AND HandsetModels IS NOT NULL;
