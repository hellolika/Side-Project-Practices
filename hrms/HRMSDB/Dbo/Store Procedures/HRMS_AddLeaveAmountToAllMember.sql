CREATE PROCEDURE [dbo].[HRMS_AddLeaveAmountToAllMember.1.0.0]
	@amount int = 1
AS
BEGIN
	DECLARE @maxAnnualLeave INT
	SELECT @maxAnnualLeave = [DefaultLeavesGranted] FROM [dbo].[LeaveType] where [TypeId] = 1;
	
	UPDATE [dbo].[LeaveAmount]
	SET [RemainingLeaves] = CASE WHEN [LeavesGranted] >= @maxAnnualLeave THEN [RemainingLeaves]
							ELSE [RemainingLeaves] + @amount END
		,[LeavesGranted] = CASE WHEN [LeavesGranted] >= @maxAnnualLeave THEN [LeavesGranted]
							ELSE [LeavesGranted] + @amount END
	WHERE [LeaveType] = 1 AND [Year] = YEAR(GETDATE())

	SELECT 0 AS ErrorCode;
END