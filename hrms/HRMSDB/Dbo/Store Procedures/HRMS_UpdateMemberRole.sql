CREATE PROCEDURE [dbo].[HRMS_UpdateMemberRole.1.0.0]
    @roleId INT,
    @memberList VARCHAR(max),
    @createdBy INT
AS
BEGIN
    SET NOCOUNT ON;

    IF NOT EXISTS(SELECT TOP 1 1 FROM [dbo].[RoleType] WHERE [Id] = @roleId AND [IsEnable] = 1)
    BEGIN
        SELECT 36 AS ErrorCode;
        RETURN;
    END

    IF NOT EXISTS(SELECT value FROM OPENJSON(@memberList) JOIN [dbo].[Member] m ON m.[Id] = value AND m.[IsDeleted] = 0)
    BEGIN
        SELECT 3 AS ErrorCode;
        RETURN;
    END
  
    DELETE FROM [dbo].[MemberRoleRecord] WHERE [RoleId] = @roleId
    INSERT INTO [dbo].[MemberRoleRecord] ([MemberId],[RoleId],[CreatedBy], [CreatedOn], [ModifiedOn])
    SELECT value, @roleId, @createdBy, GETDATE(), GETDATE() FROM OPENJSON(@memberList)
    JOIN [dbo].[Member] m ON m.[Id] = value AND m.[IsDeleted] = 0
 
    SELECT 0 AS ErrorCode;

END