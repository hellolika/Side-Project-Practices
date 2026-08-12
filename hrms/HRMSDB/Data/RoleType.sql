-- This file contains SQL statements that will be executed after the build script.
TRUNCATE TABLE [dbo].[RoleType]
GO
BEGIN
    INSERT INTO [dbo].[RoleType] (
    [RoleName],
    [RoleDescription],
    [IsEnable],
    [CreatedOn],
    [ModifiedOn]
) VALUES
('Admin', 'Manage all permission', 1, GETDATE(), GETDATE()),
('Accounting', 'Manage some permission', 1, GETDATE(), GETDATE()),
('Team Lead', 'Simple Permission', 1, GETDATE(), GETDATE()),
('HR', 'shound not see salary info', 1, GETDATE(), GETDATE()),
('Staff', 'Simple Permission', 1, GETDATE(), GETDATE()),
('Senior HR', 'Should able to see Sensitive data like salary info', 1, GETDATE(), GETDATE())
END