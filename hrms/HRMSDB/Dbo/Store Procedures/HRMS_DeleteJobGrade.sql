CREATE PROCEDURE [dbo].[HRMS_DeleteJobGrade.1.0.0]
    @jobGradeId AS INT
AS
BEGIN 
	SET NOCOUNT ON

    DECLARE @ErrorCode INT = 0;
    DECLARE @ErrorMessage NVARCHAR(255) = N'No Error';
    DECLARE @now DATETIME = GETDATE();

    BEGIN TRY
        DELETE FROM [dbo].[JobGrade]
        WHERE [Id] = @jobGradeId;
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