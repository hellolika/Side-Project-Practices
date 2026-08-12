CREATE PROCEDURE [dbo].[HRMS_DeleteMemberRole.1.0.0]
    @memberId INT,
    @roleId INT
AS
BEGIN
    SET NOCOUNT ON
    IF NOT EXISTS(SELECT TOP 1 1 FROM [dbo].[MemberRoleRecord] WHERE [MemberId] = @memberId AND [RoleId] = @roleId)
    BEGIN
        SELECT 37 AS ErrorCode;
        RETURN;
    END
  
    DELETE FROM [dbo].[MemberRoleRecord] WHERE [MemberId] = @memberId AND [RoleId] = @roleId;
    SELECT 0 AS ErrorCode;
END
GO

