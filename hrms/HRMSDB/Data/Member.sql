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
	[DepartmentId],
	[Remark],
	[TeamId],
	[WorkLocationId] 
) VALUES
('admin', 'admin@example.com', 'aMyqGvPteblK6pbcltXpU/52GQHl4PN1AOLrTQIgswAl7lLog8rCwebgCnStfBUWchKECHMF6PSpXuq/AZi/qg==', '123456789', 'PP', 300, 2, 'ABA', 0, 0, 1, 1, 1, NULL, 0),
('user', 'user@example.com', 'bGFFzGlseh+ogJSCKwDlguuxNc3iqW9TCFGcr3fosqfKhofZBQMzLWz6TraL3eXrgW4U8fbgoJS1dl6XoL+fsQ==', '123456789', 'PP', 300, 0, 'ABA', 0, 0, 1, 1, 1, NULL, 0),
('TestUser', 'testUser@example.com', 'bGFFzGlseh+ogJSCKwDlguuxNc3iqW9TCFGcr3fosqfKhofZBQMzLWz6TraL3eXrgW4U8fbgoJS1dl6XoL+fsQ==', '123456789', 'PP', 300, 0, 'ABA', 0, 0, 1, 2, 1, NULL, 0),
('Test', 'test@example.com', 'bGFFzGlseh+ogJSCKwDlguuxNc3iqW9TCFGcr3fosqfKhofZBQMzLWz6TraL3eXrgW4U8fbgoJS1dl6XoL+fsQ==', '123456789', 'PP', 300, 0, 'ABA', 0, 0, 1, 2, 1, NULL, 0),
('Test2', 'test2@example.com', 'bGFFzGlseh+ogJSCKwDlguuxNc3iqW9TCFGcr3fosqfKhofZBQMzLWz6TraL3eXrgW4U8fbgoJS1dl6XoL+fsQ==', '123456789', 'PP', 300, 0, 'ABA', 0, 0, 1, 3, 1, NULL, 0),
('Test3', 'test3@example.com', 'bGFFzGlseh+ogJSCKwDlguuxNc3iqW9TCFGcr3fosqfKhofZBQMzLWz6TraL3eXrgW4U8fbgoJS1dl6XoL+fsQ==', '123456789', 'PP', 300, 0, 'ABA', 0, 0, 1, 3, 1, NULL, 0)

END