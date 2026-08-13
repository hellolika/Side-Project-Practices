CREATE PROCEDURE [dbo].[HRMS_AddJobGrade.1.0.0]
	@jobGradeName AS NVARCHAR(255),
    @positionId AS INT,
    @createdBy AS INT
AS
BEGIN 
	SET NOCOUNT ON

    DECLARE @ErrorCode INT = 0;
    DECLARE @ErrorMessage NVARCHAR(255) = N'No Error';
    DECLARE @now DATETIME = GETDATE();

    BEGIN TRY
        INSERT INTO [dbo].[JobGrade]
        (
            [JobGradeName],
            [PositionId],
            [CreatedOn],
            [CreatedBy]
        )
        VALUES(
            @jobGradeName, 
            @positionId,
            @now, 
            @createdBy);
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