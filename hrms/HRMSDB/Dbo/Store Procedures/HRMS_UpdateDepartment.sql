CREATE PROCEDURE [dbo].[HRMS_UpdateDepartment.1.0.0]
    @departmentId AS INT,
	@departmentName AS NVARCHAR(255),
    @modifiedBy AS INT
AS
BEGIN 
	SET NOCOUNT ON

    DECLARE @ErrorCode INT = 0;
    DECLARE @ErrorMessage NVARCHAR(255) = N'No Error';
    DECLARE @now DATETIME = GETDATE();

    BEGIN TRY
        UPDATE [dbo].[Department]
        SET
            [DepartmentName] = @departmentName, 
            [ModifiedOn] = @now, 
            [ModifiedBy] = @modifiedBy
        WHERE [Id] = @departmentId;
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