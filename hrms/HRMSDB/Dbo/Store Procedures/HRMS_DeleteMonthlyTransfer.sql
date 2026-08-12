CREATE PROCEDURE [dbo].[HRMS_DeleteMonthlyTransfer.1.0.0]
  @transferId INT
AS

IF NOT EXISTS(SELECT TOP 1 1 FROM [dbo].[Transfer] WHERE [Id] = @transferId)
	BEGIN
		SELECT 31 AS ErrorCode;
	END
	ELSE
	BEGIN
		DELETE FROM [dbo].[Transfer] WHERE Id = @transferId
		SELECT 0 AS ErrorCode;
	END