CREATE PROCEDURE [dbo].[HRMS_AddTeam.1.0.0]
	@teamName AS NVARCHAR(255),
    @departmentId AS INT,
    @startTime AS TIME(0),
    @endTime AS TIME(0),
    @totalHour AS DECIMAL(16,9),
    @createdBy AS INT
AS
BEGIN 
	SET NOCOUNT ON

    DECLARE @ErrorCode INT = 0;
    DECLARE @ErrorMessage NVARCHAR(255) = N'No Error';
    DECLARE @now DATETIME = GETDATE();

    BEGIN TRY
        INSERT INTO [dbo].[Team]
        (
            [TeamName],
            [DepartmentId],
            [StartTime],
            [EndTime],
            [TotalHour],
            [CreatedOn],
            [CreatedBy]
        )
        VALUES(
            @teamName,
            @departmentId,
            @startTime,
            @endTime,
            @totalHour,
            @now,
            @createdBy
        );
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