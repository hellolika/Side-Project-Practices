CREATE PROCEDURE [dbo].[HRMS_PopulateMemberRoleRecord.1.0.0]
    @roleId INT
AS
BEGIN 
	SET NOCOUNT ON

    INSERT INTO [dbo].[MemberRoleRecord](
        [MemberId],
        [RoleId],
        [CreatedOn],
        [ModifiedOn]
    )
    SELECT m.Id, @roleId, GETDATE(), GETDATE()
    FROM [dbo].[Member] m
    WHERE  m.[IsResigned] = 0 AND 
    NOT EXISTS(SELECT TOP 1 1 FROM [dbo].[MemberRoleRecord] mr WITH(NOLOCK)
    WHERE m.Id = mr.MemberId)

    SELECT 0 AS ErrorCode;

END