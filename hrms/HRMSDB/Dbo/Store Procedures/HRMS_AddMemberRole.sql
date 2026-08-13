CREATE PROCEDURE [dbo].[HRMS_AddMemberRole.1.0.0]
    @memberId INT,
    @roleId INT,
    @createdBy INT
AS
BEGIN
	SET NOCOUNT ON;
    
    IF NOT EXISTS(SELECT TOP 1 1 FROM [dbo].[Member] WHERE [Id] = @memberId AND [IsDeleted] = 0)
        BEGIN
            SELECT 3 AS ErrorCode;
            RETURN;
        END

    IF NOT EXISTS(SELECT TOP 1 1 FROM [dbo].[Member] WHERE [Id] = @createdBy AND [IsDeleted] = 0)
        BEGIN
            SELECT 3 AS ErrorCode;
            RETURN;
        END

    IF NOT EXISTS(SELECT TOP 1 1 FROM [dbo].[RoleType] WHERE [Id] = @roleId AND [IsEnable] = 1)
        BEGIN
            SELECT 36 AS ErrorCode;
            RETURN;
        END
    
    IF EXISTS(SELECT TOP 1 1 FROM [dbo].[MemberRoleRecord] WHERE [MemberId] = @memberId AND [RoleId] = @roleId)
        BEGIN
            SELECT 35 AS ErrorCode;
            RETURN;
        END

    INSERT INTO [dbo].[MemberRoleRecord]
    ([MemberId], [RoleId],[CreatedBy], [CreatedOn], [ModifiedOn])
    VALUES(@memberId, @roleId, @createdBy, GETDATE(), GETDATE());
    
    SELECT 0 AS ErrorCode;

END