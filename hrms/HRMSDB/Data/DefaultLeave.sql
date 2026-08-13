TRUNCATE TABLE [dbo].[DefaultLeave]
GO

BEGIN
	INSERT INTO [dbo].[DefaultLeave] (
	[LeaveType],
	[Amount]
) VALUES
(1, 12),
(2, 12)

END