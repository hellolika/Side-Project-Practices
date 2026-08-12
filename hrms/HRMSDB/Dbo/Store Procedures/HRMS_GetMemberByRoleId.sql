CREATE PROCEDURE [dbo].[HRMS_GetMemberByRoleId.1.0.0]
 @roleId INT
AS
BEGIN 
	SET NOCOUNT ON
	SELECT mrr.[MemberId], m.[Username], m.[Email]
	FROM [dbo].[MemberRoleRecord] mrr
	JOIN [dbo].[Member] m ON m.[Id] = mrr.[MemberId] AND m.[IsDeleted] = 0 
	WHERE mrr.[RoleId] = @roleId
END
GO