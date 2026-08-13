CREATE PROCEDURE [dbo].[HRMS_CancelLeave.1.0.0]
	@memberId INT,
	@requestId INT
AS
BEGIN
	SET NOCOUNT ON;

	IF EXISTS (SELECT TOP 1 1 FROM [dbo].[TakeLeaveRecord] WITH(NOLOCK) 
	WHERE [RequestId] = @requestId AND [MemberId] = @memberId)
	BEGIN
		IF EXISTS (SELECT TOP 1 1 FROM [dbo].[TakeLeaveRecord] WITH(NOLOCK) 
					WHERE [RequestId] = @requestId AND [Status] IN (0,1) 
					AND [IsCancel] = 0 AND [MemberId] = @memberId)
		BEGIN
			DECLARE @leaveAmount AS DECIMAL(16,9)
			DECLARE @leaveType AS INT
			DECLARE @year AS INT

			UPDATE [dbo].[TakeLeaveRecord]
			SET [IsCancel] = 1,
			@leaveAmount = [NumberOfDay],
			@leaveType = [LeaveType],
			@year = YEAR([StartDate])
			WHERE [RequestId] = @requestId
			AND [Status] IN (0, 1) 
			AND [IsCancel] = 0;

			UPDATE [dbo].[LeaveAmount]
			SET [RemainingLeaves] = [RemainingLeaves] + @leaveAmount
			WHERE [MemberId] = @memberId 
			AND [Year] = @year AND [LeaveType] = @leaveType;

			SELECT 0 AS ErrorCode;
		END
		ELSE
		BEGIN
			SELECT 12 AS ErrorCode;
		END
	END
	ELSE
	BEGIN
		SELECT 11 AS ErrorCode;
	END
END
