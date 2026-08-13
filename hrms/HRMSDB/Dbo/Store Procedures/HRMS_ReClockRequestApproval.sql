CREATE PROCEDURE [dbo].[HRMS_ReClockRequestApproval.1.0.0]
	@approverId INT,
	@requestId INT,
	@isApproved BIT,
	@responseReason nvarchar(200)
AS
BEGIN
	SET NOCOUNT ON;
	IF EXISTS (SELECT TOP 1 1 FROM [dbo].[ReClockRecord] WITH(NOLOCK) WHERE [RequestId] = @RequestId)
	BEGIN
		UPDATE [dbo].[ReClockRecord] 
		SET [Status] = IIF(@isApproved = 1, 1, 2),
		[UpdateBy] = @approverId, [ResponseReason] = @responseReason
		WHERE [RequestId] = @requestId;
		SELECT 0 AS ErrorCode;
	END
	ELSE
	BEGIN
		SELECT 11 AS ErrorCode;
	END
END
