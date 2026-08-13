TRUNCATE TABLE [dbo].[LeaveType]
GO

BEGIN
	INSERT INTO [dbo].[LeaveType] (
	[Type],
	[DefaultLeavesGranted],
	[IsEnable],
	[IsLimited]
) VALUES
('Unpaid Leave', 0, 1, 0)
,('Annual Leave', 12, 1, 1)
,('Sick Leave', 10, 1, 1)
,('Wedding Leave',7, 1, 1)
,('Duty Leave', 12, 1, 1)
,('Paternity Leave', 7, 1, 1)

END