CREATE PROCEDURE [dbo].[HRMS_GetMemberLeaveAmountV2.1.0.0]
	@memberId INT
AS
BEGIN
	SET NOCOUNT ON;

	SELECT la.[LeaveType] AS [LeaveId]
		,lt.[Type] AS [LeaveType]
		,la.[RemainingLeaves] AS [Availability]
		,la.[LeavesGranted] AS [Earned]
		,(la.[LeavesGranted] - la.[RemainingLeaves]) AS [Taken]
		,lt.[DefaultLeavesGranted] AS [Allowance] 
		,la.[Year] 
	FROM [dbo].[LeaveAmount] la WITH(NOLOCK) 
	INNER JOIN [dbo].[LeaveType] lt WITH(NOLOCK)
		ON la.[LeaveType] = lt.[TypeId]
	WHERE la.[MemberId] = @memberId
		AND la.[Year] = YEAR(GETDATE())
END