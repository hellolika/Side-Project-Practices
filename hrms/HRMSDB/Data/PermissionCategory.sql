TRUNCATE TABLE [dbo].[PermissionCategory]
GO
BEGIN
	INSERT INTO [dbo].[PermissionCategory] (
	[PermissionCategoryName],
		[IsEnable],
		[CreatedOn],
		[ModifiedOn]
	) VALUES
		('Member', 1, GETDATE(), GETDATE()), 
		('Payroll', 1, GETDATE(), GETDATE()),
		('Leave', 1, GETDATE(), GETDATE()),
		('Attendance', 1, GETDATE(), GETDATE()),
		('Announcement', 1, GETDATE(), GETDATE()),
		('Role', 1, GETDATE(), GETDATE()),
		('Permission', 1, GETDATE(), GETDATE()),
		('Clock', 1, GETDATE(), GETDATE()),
		('Location',1,GETDATE(),GETDATE()),
		('SystemManagement',1,GETDATE(),GETDATE())
END
