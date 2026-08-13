CREATE PROCEDURE [dbo].[HRMS_GetRoleByMemberId.1.0.0]
    @memberId INT
AS
BEGIN 

	SET NOCOUNT ON
    SELECT rt.[Id],[RoleName],[RoleDescription]
    FROM [dbo].[MemberRoleRecord] mrr
    JOIN [dbo].[RoleType] rt on rt.[Id] = mrr.[RoleId]
    WHERE mrr.[MemberId] = @memberId
END