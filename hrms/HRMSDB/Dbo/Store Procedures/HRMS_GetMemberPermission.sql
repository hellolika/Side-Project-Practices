CREATE PROCEDURE [dbo].[HRMS_GetMemberPermission.1.0.0]
    @memberId INT 
AS
BEGIN
    SET NOCOUNT ON;

    SELECT mr.[RoleId], rt.[RoleName] as RoleName, rp.[PermissionId] as PermissionId , pt.[PermissionName]  as PermissionName
    FROM [dbo].[MemberRoleRecord] mr
    JOIN [dbo].[RolePermissionRecord] rp ON mr.[RoleId] = rp.[RoleId]
    JOIN [dbo].[PermissionType] pt ON pt.[Id] = rp.[PermissionId]
    JOIN [dbo].[RoleType] rt ON rt.[Id] = mr.[RoleId]
    WHERE mr.[MemberId] = @memberId AND rt.[IsEnable] = 1

END


