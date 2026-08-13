CREATE PROCEDURE [dbo].[HRMS_UpdateLocation.1.0.0]
    @locationId AS INT,
    @locationName AS NVARCHAR(255),
    @latitude AS NVARCHAR(255),
    @longitude AS NVARCHAR(255),
    @range AS INT,
    @isEnabled AS BIT,
    @modifiedBy AS INT
AS
BEGIN 
	SET NOCOUNT ON

    DECLARE @ErrorCode INT = 0;
    DECLARE @ErrorMessage NVARCHAR(255) = N'No Error';
    DECLARE @now DATETIME = GETDATE();

    BEGIN TRY
        UPDATE [dbo].[LocationDetail]
        SET
            [LocationName] = @locationName, 
            [Latitude] = @latitude, 
            [Longitude] = @longitude, 
            [Range] = @range, 
            [IsEnabled] = @isEnabled, 
            [ModifiedOn] = @now, 
            [ModifiedBy] = @modifiedBy
        WHERE [Id] = @locationId;
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