CREATE PROCEDURE [dbo].[HRMS_BatchUpdateMonthlyTransferStatus.1.0.0]
    @statusId INT,
    @transferIdList VARCHAR(max)
AS
BEGIN
    SET NOCOUNT ON;
    UPDATE [dbo].[Transfer] SET [Status] = @statusId WHERE Id IN (SELECT value FROM OPENJSON(@transferIdList));   
    SELECT 0 AS ErrorCode
END