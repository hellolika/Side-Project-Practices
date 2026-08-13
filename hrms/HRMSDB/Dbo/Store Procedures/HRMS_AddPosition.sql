CREATE PROCEDURE [dbo].[HRMS_AddPosition.1.0.0]
	@positionName AS NVARCHAR(255),
    @teamId AS INT,
    @createdBy AS INT
AS
BEGIN 
	SET NOCOUNT ON

    DECLARE @ErrorCode INT = 0;
    DECLARE @ErrorMessage NVARCHAR(255) = N'No Error';
    DECLARE @now DATETIME = GETDATE();

    BEGIN TRY
        INSERT INTO [dbo].[Position]
        (
            [PositionName], 
            [TeamId],
            [CreatedOn],
            [CreatedBy]
        )
        VALUES(
            @positionName,
            @teamId, 
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