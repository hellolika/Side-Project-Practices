CREATE PROCEDURE [dbo].[HRMS_UpdateRoleByMemberId.1.0.0]
    @memberId INT,
    @roleList VARCHAR(max),
    @createdBy INT
AS
BEGIN

    SET NOCOUNT ON;

    IF NOT EXISTS(SELECT TOP 1 1 FROM [dbo].[Member] WHERE [Id] = @memberId AND [IsDeleted] = 0)
        BEGIN
            SELECT 3 AS ErrorCode;
            RETURN;
        END

    IF EXISTS (SELECT value FROM OPENJSON(@roleList)
    LEFT JOIN [dbo].[RoleType] rt ON rt.[Id] = value
    WHERE rt.[Id] is NULL)
    BEGIN
        SELECT 36 AS ErrorCode;
        RETURN;
    END

    DELETE FROM [dbo].[MemberRoleRecord] WHERE [MemberId] = @memberId;
    INSERT INTO [dbo].[MemberRoleRecord]
	([MemberId], [RoleId],[CreatedBy], [CreatedOn], [ModifiedOn])
    SELECT @memberId, value,@createdBy, GETDATE(), GETDATE() FROM OPENJSON(@roleList)

    SELECT 0 AS ErrorCode;

END