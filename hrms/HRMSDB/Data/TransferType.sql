TRUNCATE TABLE [dbo].[TransferType]
GO 
BEGIN

	INSERT INTO [dbo].[TransferType] (
	[TransferName],[BeneficiaryTypeId]
	) VALUES
('Monthly Salary',0),
('Unpaid Leave',2),
('Traffic Subsidy',1),
('Rental Subsidy',1),
('Food Subsidy',1),
('Team Leader Subsidy',1),
('Support Subsidy',1),
('Travel Subsidy',1),
('Equipment Subsidy',1),
('Non-working',2)
END
