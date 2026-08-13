CREATE PROCEDURE [dbo].[HRMS_GetPermissionByRoleId.1.0.0]
    @roleId INT
AS
BEGIN 
	SET NOCOUNT ON
    SELECT [PermissionId]
    FROM [dbo].[RolePermissionRecord] rpr
    WHERE rpr.[RoleId] = @roleId	
END