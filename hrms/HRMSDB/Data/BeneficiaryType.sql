-- This file contains SQL statements that will be executed after the build script.
TRUNCATE TABLE [dbo].[BeneficiaryType]
GO 
BEGIN
	INSERT INTO [dbo].[BeneficiaryType] (
	    [Name],[CreatedOn]
	) VALUES
('Allowance',GETDATE()),
('Deduction',GETDATE())
END

