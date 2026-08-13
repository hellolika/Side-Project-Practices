CREATE PROCEDURE [dbo].[HRMS_GetAllPermission.1.0.0]
AS
BEGIN 
	SET NOCOUNT ON
    SELECT pt.[Id], pt.[PermissionName], pc.[PermissionCategoryName]
    FROM [dbo].[PermissionType] pt
    JOIN [dbo].[PermissionCategory] pc ON pc.[Id] = pt.[PermissionCategory]
    WHERE pt.[IsEnable] = 1
    GROUP BY pc.[PermissionCategoryName], pt.[Id] , pt.[PermissionName]
 
END