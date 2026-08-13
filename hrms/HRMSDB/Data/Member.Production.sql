TRUNCATE TABLE [dbo].[Member]
GO

BEGIN
	INSERT INTO [dbo].[Member] (
	[Username],
	[Email],
	[Password],
	[PhoneNumber],
	[Address],
	[Salary],
	[Permission],
	[BankAccount],
	[IsInProbation],
	[IsResigned],
	[Department],
	[Remark],
	[TeamId],
	[WorkLocationId] 
) VALUES
('admin', 'admin@example.com', 'aMyqGvPteblK6pbcltXpU/52GQHl4PN1AOLrTQIgswAl7lLog8rCwebgCnStfBUWchKECHMF6PSpXuq/AZi/qg==', '123456789', 'PP', 300, 2, 'ABA', 0, 0, 1, 1, 1, NULL, 0)

END