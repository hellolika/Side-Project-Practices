CREATE PROCEDURE [dbo].[HRMS_GetLeaveRecordsByMemberId.3.0.0]
	@memberId INT
AS
BEGIN
	SET NOCOUNT ON;

	SELECT t.[RequestId], t.[LeaveType] AS [LeaveId], l.[Type] AS [LeaveType], 
	t.[NumberOfDay], t.[StartDate], t.[EndDate], t.[IsCancel], t.[Status], t.[Image],
	t.[Reason], t.[ResponseReason], t.[SubmittedOn], ub.[Username] AS [Approver]
	FROM [dbo].[TakeLeaveRecord] t WITH(NOLOCK) 
	INNER JOIN [dbo].[LeaveType] l WITH(NOLOCK) ON t.[LeaveType] = l.[TypeId]
	LEFT JOIN [dbo].[Member] ub WITH(NOLOCK) ON ub.[Id] = t.[UpdateBy]
	WHERE [MemberId] = @memberId
	ORDER BY [StartDate] DESC;
END