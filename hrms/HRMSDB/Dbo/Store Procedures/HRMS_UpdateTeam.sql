CREATE PROCEDURE [dbo].[HRMS_UpdateTeam.1.0.0]
    @teamId AS INT,
	@teamName AS NVARCHAR(255),
    @departmentId AS INT,
    @startTime AS TIME(0),
    @endTime AS TIME(0),
    @totalHour AS DECIMAL(16,9),
    @modifiedBy AS INT
AS
BEGIN 
	SET NOCOUNT ON

    DECLARE @ErrorCode INT = 0;
    DECLARE @ErrorMessage NVARCHAR(255) = N'No Error';
    DECLARE @now DATETIME = GETDATE();

    BEGIN TRY
        UPDATE [dbo].[Team]
        SET
            [TeamName] = @teamName,
            [DepartmentId] = @departmentId,
            [StartTime] = @startTime,
            [EndTime] = @endTime,
            [TotalHour] = @totalHour,
            [ModifiedOn] = @now,
            [ModifiedBy] = @modifiedBy
        WHERE [TeamId] = @teamId;
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