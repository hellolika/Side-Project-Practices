-- -- This file contains SQL statements that will be executed after the build script.
BEGIN
    SET NOCOUNT ON;

     -- This group of statement is use to populate default super admin role permission
    -- which permission contain all permissions
    -- RoleId = 1 is Super-admin Role

    INSERT INTO [dbo].[RolePermissionRecord]
    ([RoleId],[PermissionId],[CreatedBy],[CreatedOn],[ModifiedOn])
    SELECT 1, pt.[Id],1,GETDATE(),GETDATE()
    FROM [PermissionType] pt
    LEFT JOIN [RolePermissionRecord] rpr ON rpr.[PermissionId] = pt.[Id] AND rpr.[RoleId] = 1
    WHERE rpr.[PermissionId] IS NULL

    -- IF NOT EXISTS(SELECT TOP 1 1 FROM [dbo].[RolePermissionRecord] WHERE [RoleId] = 1)
    -- BEGIN
    --     INSERT INTO [dbo].[RolePermissionRecord](
    --     [RoleId],
    --     [PermissionId],
    --     [CreatedOn],
    --     [ModifiedOn]
    --     )
    --     SELECT 1, pt.[Id], GETDATE(), GETDATE()
    --     FROM [dbo].[PermissionType] pt
    -- END
    -- INSERT INTO [dbo].[RolePermissionRecord](
    --     [RoleId],
    --     [PermissionId],
    --     [CreatedOn],
    --     [ModifiedOn]
    -- )VALUES
    -- (5, 52, GETDATE(),GETDATE()),
    -- (5, 50, GETDATE(),GETDATE()),
    -- (5,47, GETDATE(),GETDATE()),
    -- (5, 56, GETDATE(),GETDATE()),
    -- (5, 55, GETDATE(),GETDATE()),
    -- (5, 40, GETDATE(),GETDATE()),
    -- (5, 41, GETDATE(),GETDATE()),
    -- (5, 49, GETDATE(),GETDATE()),
    -- (5, 51, GETDATE(),GETDATE()),
    -- (5, 43, GETDATE(),GETDATE()),
    -- (5, 48, GETDATE(),GETDATE()),
    -- (5, 45, GETDATE(),GETDATE()),
    -- (5, 44, GETDATE(),GETDATE()),
    -- (5, 46, GETDATE(),GETDATE()),
    -- (5, 41, GETDATE(),GETDATE()),
    -- (5, 54, GETDATE(),GETDATE()),
    -- (5, 53, GETDATE(),GETDATE())

    -- -- This group of statement is use to populate default employee role permission
    -- -- which permission contain permission category 2(Member) and 3(Data)
    -- -- RoleId = 3 is Employee Role
    -- DELETE FROM [dbo].[RolePermissionRecord] WHERE [RoleId] = 3;
    -- INSERT INTO [dbo].[RolePermissionRecord](
    --     [RoleId],
    --     [PermissionId],
    --     [CreatedOn],
    --     [ModifiedOn]
    -- )
    -- SELECT 3, pt.[Id], GETDATE(), GETDATE()
    -- FROM [dbo].[PermissionType] pt WHERE pt.[PermissionCategory] = 2 OR pt.[PermissionCategory] = 3

END