CREATE PROCEDURE [dbo].[HRMS_UpsertSlackUsers.1.0.0]
    @slackUserData SlackUserDataType READONLY
AS
BEGIN
    SET NOCOUNT ON;

    DECLARE @ErrorCode INT = 0;
    DECLARE @ErrorMessage NVARCHAR(255) = N'No Error';

    DECLARE @Now DATETIME = GETDATE();

    BEGIN TRY
        MERGE [dbo].[SlackUserInfo] AS target
        USING @slackUserData AS source
        ON target.Id = source.Id
        WHEN MATCHED
        THEN
            UPDATE SET
                [Username] = source.Username,
                [RealName] = source.RealName,
                [Email] = source.Email
        WHEN NOT MATCHED
        THEN
            INSERT ([Id], [Username], [RealName], [Email])
            VALUES (source.Id, source.Username, source.RealName, source.Email);
    END TRY
    BEGIN CATCH
        SET @ErrorCode = ERROR_NUMBER();
        SET @ErrorMessage = ERROR_MESSAGE();
    END CATCH

    IF @ErrorCode <> 0
    BEGIN
        SELECT @ErrorCode AS ErrorCode, @ErrorMessage AS ErrorMessage;
    END
    ELSE
    BEGIN
        SELECT 0 AS ErrorCode, 'Success' AS ErrorMessage;
    END
END