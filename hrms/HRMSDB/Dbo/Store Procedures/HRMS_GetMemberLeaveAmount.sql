CREATE PROCEDURE [dbo].[HRMS_GetMemberLeaveAmount.2.0.0]
	@memberId INT
AS
BEGIN
	SET NOCOUNT ON;

	SELECT la.[LeaveType] AS [LeaveId], lt.[Type] AS [LeaveType], 
	la.[RemainingLeaves], la.[LeavesGranted], la.[Year] 
	FROM [dbo].[LeaveAmount] la WITH(NOLOCK) 
	INNER JOIN [dbo].[LeaveType] lt WITH(NOLOCK) ON la.[LeaveType] = lt.[TypeId]
	WHERE la.[MemberId] = @memberId AND la.[Year] = YEAR(GETDATE())
END