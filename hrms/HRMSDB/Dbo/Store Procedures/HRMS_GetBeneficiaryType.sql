CREATE PROCEDURE [dbo].[HRMS_GetBeneficiaryType.1.0.0]
AS
BEGIN 
    SELECT [Id], [Name] FROM [dbo].[BeneficiaryType]
END