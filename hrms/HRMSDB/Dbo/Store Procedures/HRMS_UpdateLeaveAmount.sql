CREATE PROCEDURE [dbo].[HRMS_UpdateLeaveAmount.2.0.0]
@memberId INT,
@leaveAmount AS DECIMAL(16,9),
@leaveType INT,
@year INT

AS
BEGIN
	SET NOCOUNT ON;
	IF NOT EXISTS (SELECT 1 FROM [dbo].[Member] WHERE [Id] = @memberId)
	BEGIN
		SELECT 3 AS ErrorCode;
		RETURN;
	END

	IF NOT EXISTS (SELECT 1 FROM [dbo].[LeaveType] WHERE [TypeId] = @leaveType AND [IsEnable] = 1) OR
	NOT EXISTS (SELECT 1 FROM [dbo].[LeaveAmount] WHERE [MemberId] = @memberId AND [LeaveType] = @leaveType AND [Year] = @year)
	BEGIN
		SELECT 13 AS ErrorCode;
		RETURN;
	END

	UPDATE [dbo].[LeaveAmount]
	SET	[RemainingLeaves] = [RemainingLeaves] - @leaveAmount,
		[LeavesGranted] = @leaveAmount,
		[ModifyOn] = GETDATE()
	WHERE [MemberId] = @memberId AND [LeaveType] = @leaveType AND [Year] = @year

	SELECT 0 AS ErrorCode;

END