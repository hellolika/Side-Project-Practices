--TRUNCATE TABLE [dbo].[LeaveAmount]
--GO

BEGIN
	DECLARE @currentYear AS INT = YEAR(GETDATE());

	INSERT INTO [dbo].[LeaveAmount] (
	[MemberId], [LeaveType], [RemainingLeaves], [LeavesGranted], [Year]
	)
	SELECT m.Id, lt.TypeId, lt.DefaultLeavesGranted, lt.DefaultLeavesGranted, @currentYear
	FROM [dbo].[Member] m WITH(NOLOCK), [dbo].[LeaveType] lt WITH(NOLOCK)
	WHERE lt.[IsEnable] = 1 AND lt.[IsLimited] = 1 AND
	NOT EXISTS(SELECT TOP 1 1 FROM [dbo].[LeaveAmount] la WITH(NOLOCK) 
	WHERE la.[MemberId] = m.[Id] AND la.[LeaveType] = lt.[TypeId] 
	AND la.[Year] = @currentYear)

END