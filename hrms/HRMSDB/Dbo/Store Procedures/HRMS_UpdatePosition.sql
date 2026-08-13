CREATE PROCEDURE [dbo].[HRMS_UpdatePosition.1.0.0]
    @positionId AS INT,
	@positionName AS NVARCHAR(255),
    @teamId AS INT,
    @modifiedBy AS INT
AS
BEGIN 
	SET NOCOUNT ON

    DECLARE @ErrorCode INT = 0;
    DECLARE @ErrorMessage NVARCHAR(255) = N'No Error';
    DECLARE @now DATETIME = GETDATE();

    BEGIN TRY
        UPDATE [dbo].[Position]
        SET
            [PositionName] = @positionName,
            [TeamId] = @teamId,
            [ModifiedOn] = @now,
            [ModifiedBy] = @modifiedBy
        WHERE [Id] = @positionId;
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