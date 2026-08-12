CREATE PROCEDURE [dbo].[HRMS_GetAllLeaveAmount.2.0.0]
AS
BEGIN
	SET NOCOUNT ON;

	SELECT la.[MemberId], m.[Username], la.[LeaveType] AS [LeaveId], 
	lt.[Type] AS [LeaveType], la.[RemainingLeaves], la.[LeavesGranted],
	la.[Year], la.[ModifyOn], la.[ModifyBy]
	FROM [dbo].[LeaveAmount] la WITH(NOLOCK)
	JOIN [dbo].[LeaveType] lt WITH(NOLOCK) ON la.[LeaveType] = lt.[TypeId]
	JOIN [dbo].[Member] m ON la.[MemberId] = m.[Id]
	WHERE lt.[IsEnable] = 1 AND 
	m.[IsResigned] = 0

END