CREATE PROCEDURE [dbo].[HRMS_UpdateRolePermission.1.0.0]
    @roleId INT,
    @permissionList VARCHAR(max)
AS
BEGIN
	SET NOCOUNT ON;

    IF NOT EXISTS(SELECT TOP 1 1 FROM [dbo].[RoleType] WHERE [Id] = @roleId AND [IsEnable] = 1)
    BEGIN
        SELECT 36 AS ErrorCode;
        RETURN;
    END

    DELETE FROM [dbo].[RolePermissionRecord] WHERE [RoleId] = @roleId;
    INSERT INTO [dbo].[RolePermissionRecord] ([RoleId],[PermissionId],[CreatedOn],[ModifiedOn])
    SELECT @roleId, value, GETDATE(), GETDATE() FROM OPENJSON(@permissionList)
    JOIN [dbo].[PermissionType] pt ON pt.[Id] = value

    SELECT 0 AS ErrorCode

END